# Copyright (c) 2025 by Robert Senatorov. All rights reserved.
# train.py
"""
Transformer Chatbot Training Script
------------------------------------
This script trains a Transformer-based chatbot model on pre-tokenized Q&A pairs.
The model has 12 layers, a 768-dimensional embedding, 12 attention heads, and a
feedforward dimension of 3072. It uses 0.1 dropout and mixed precision (FP16 with TF32).
The effective global batch size is 256 (achieved via micro-batches of 128 and grad_accum_steps=2).
Training is performed for exactly 3 epochs.
The final model is saved as rpt1.pth and rpt1.pt, and loss statistics are saved as PNG, CSV, and TXT files.
"""

#######################
#    Warning Setup    #
#######################
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings in production

#######################
#      Imports        #
#######################
import os
import sys
import json
import math
import random
import logging
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from functools import partial
from tqdm import tqdm
import psutil
import sentencepiece as spm

import matplotlib.pyplot as plt

# Enable TF32 for faster FP32 matrix multiplication on Ampere/Lovelace GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#######################
#  Configuration      #
#######################
CONFIG = {
    "train_split": 0.9,
    "num_epochs": 3,
    "batch_size": 128,
    "grad_accum_steps": 2,
    "max_seq_len": 128,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "checkpoint_dir": "checkpoints",
    "logs_dir": "logs",
    "token_file": "dataset/out_tokens.jsonl",
    "sp_model_file": "dataset/vocab/tokenizer.model",
    "sp_vocab_file": "dataset/vocab/tokenizer.vocab",
    "device": "cuda",
    "use_amp": True,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "num_workers": 4,
    "pin_memory": False,
    "test_pipeline": False
}

# Special token IDs for SentencePiece
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
USR_TOKEN = "<usr>"
COT_TOKEN = "<cot>"
BOT_TOKEN = "<bot>"
UNK_TOKEN = "<unk>"

#######################
#  Directory Setup    #
#######################
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["logs_dir"], exist_ok=True)

#######################
#   Logging Setup     #
#######################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(CONFIG["logs_dir"], "training.log"))
    ]
)
logger = logging.getLogger(__name__)

##############################
#   Dataset Definition       #
##############################
class ChatbotDataset(Dataset):
    """
    Loads all pre-tokenized Q&A pairs from a JSONL file into memory.
    Each line is expected to be a JSON object with a key "tokens" containing a list of integers.
    """
    def __init__(self, token_file: str):
        super().__init__()
        self.samples = []
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj["tokens"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, max_seq_len: int, pad_token_id: int):
    """
    Prepares input-target pairs from sequences padded to max_seq_len.
    """
    inputs, targets = [], []
    for tokens in batch:
        if len(tokens) != max_seq_len:
            tokens = tokens[:max_seq_len]
            if len(tokens) < max_seq_len:
                tokens += [pad_token_id] * (max_seq_len - len(tokens))
        if len(tokens) < 2:
            continue
        x = tokens[:-1]
        y = tokens[1:]
        inputs.append(x)
        targets.append(y)
    x_tensor = torch.tensor(inputs, dtype=torch.long)
    y_tensor = torch.tensor(targets, dtype=torch.long)
    return x_tensor, y_tensor

#########################################
#  Vocabulary and Tokenizer Loading     #
#########################################
def load_tokenizer(sp_model_file: str):
    """Load the SentencePiece tokenizer and determine special token IDs."""
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(sp_model_file)
    
    pad_token_id = tokenizer.piece_to_id(PAD_TOKEN)
    eos_token_id = tokenizer.piece_to_id(EOS_TOKEN)
    usr_token_id = tokenizer.piece_to_id(USR_TOKEN)
    cot_token_id = tokenizer.piece_to_id(COT_TOKEN)
    bot_token_id = tokenizer.piece_to_id(BOT_TOKEN)
    unk_token_id = tokenizer.piece_to_id(UNK_TOKEN)
    
    if pad_token_id == -1:
        pad_token_id = 0
    if eos_token_id == -1:
        eos_token_id = 2
    if unk_token_id == -1:
        unk_token_id = 1
    
    logger.info(f"Special token IDs: PAD={pad_token_id}, EOS={eos_token_id}, UNK={unk_token_id}, "
                f"USR={usr_token_id}, COT={cot_token_id}, BOT={bot_token_id}")
    
    vocab_size = tokenizer.get_piece_size()
    return tokenizer, vocab_size, pad_token_id, eos_token_id

##############################
#     Model Definition       #
##############################
class TransformerDecoder(nn.Module):
    """
    Decoder-only Transformer for next-token prediction.
    Architecture details:
      - Embedding dimension: CONFIG["embed_dim"]
      - Number of layers: CONFIG["num_layers"]
      - Number of attention heads: CONFIG["num_heads"]
      - Feedforward dimension: embed_dim * 4
      - Dropout: CONFIG["dropout"]
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(positions)
        hidden_states = tok_emb + pos_emb
        hidden_states = self.dropout(hidden_states)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device) == 1, diagonal=1)
        causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))

        memory = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, memory, tgt_mask=causal_mask)
        hidden_states = self.ln_f(hidden_states)
        logits = self.fc_out(hidden_states)
        return logits

def get_loss_fn(pad_token_id, label_smoothing):
    """
    Returns the CrossEntropyLoss function with label smoothing and ignore_index for PAD tokens.
    """
    return nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=label_smoothing)

##############################
#    Main Training Loop      #
##############################
def train():
    # Set seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    logger.info("Training configuration:")
    logger.info(f"  Model dimensions: {CONFIG['embed_dim']}")
    logger.info(f"  Number of layers: {CONFIG['num_layers']}")
    logger.info(f"  Number of heads: {CONFIG['num_heads']}")
    effective_batch = CONFIG["batch_size"] * CONFIG["grad_accum_steps"]
    logger.info(f"  Effective batch size: {effective_batch}")
    logger.info(f"  Sequence length: {CONFIG['max_seq_len']}")
    logger.info(f"  Training for {CONFIG['num_epochs']} epochs")

    tokenizer, vocab_size, pad_token_id, eos_token_id = load_tokenizer(CONFIG["sp_model_file"])
    logger.info(f"Vocabulary size: {vocab_size} (PAD id: {pad_token_id}, EOS id: {eos_token_id})")

    full_dataset = ChatbotDataset(CONFIG["token_file"])
    total_samples = len(full_dataset)
    indices = np.random.permutation(total_samples)
    
    if CONFIG["test_pipeline"]:
        train_size = max(1, int(total_samples * 0.01))
        val_size = max(1, int(total_samples * 0.01))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        logger.info(f"Test pipeline mode: Using {len(train_indices)} training samples and {len(val_indices)} validation samples out of {total_samples} total.")
    else:
        train_size = int(total_samples * CONFIG["train_split"])
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    logger.info(f"Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn, max_seq_len=CONFIG["max_seq_len"], pad_token_id=pad_token_id),
        drop_last=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"]
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=CONFIG["batch_size"] * 2,
        shuffle=False,
        collate_fn=partial(collate_fn, max_seq_len=CONFIG["max_seq_len"], pad_token_id=pad_token_id),
        drop_last=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"]
    )

    model = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        max_seq_len=CONFIG["max_seq_len"]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params / 1e6:.2f}M parameters")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    logger.info("Using standard AdamW optimizer.")

    total_steps = (len(train_loader) // CONFIG["grad_accum_steps"]) * CONFIG["num_epochs"]
    logger.info(f"Total training steps: {total_steps}")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    criterion = get_loss_fn(pad_token_id, CONFIG["label_smoothing"])
    scaler = torch.amp.GradScaler(enabled=CONFIG["use_amp"] and device.type == "cuda")

    global_step = 0
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(CONFIG["logs_dir"], "tensorboard"))
        have_tensorboard = True
    except Exception:
        have_tensorboard = False

    overall_pbar = tqdm(total=total_steps, desc="Overall progress", position=0, leave=True)
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()
        # Training progress bar for the epoch
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Training", position=1, leave=False)
        for i, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=CONFIG["use_amp"] and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (i + 1) % CONFIG["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                overall_pbar.update(1)

                if global_step % 10 == 0:
                    avg_loss = running_loss / (i + 1)
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({'Train Loss': f"{avg_loss:.4f}", 'LR': f"{current_lr:.4e}", 'Step': global_step})
                    if have_tensorboard:
                        tb_writer.add_scalar('Loss/train', avg_loss, global_step)
                        tb_writer.add_scalar('LearningRate', current_lr, global_step)

        train_loss_epoch = running_loss / len(train_loader)
        # Validation with progress bar
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc="Epoch {} Validation".format(epoch), position=1, leave=False)
        with torch.no_grad():
            for x_val, y_val in val_pbar:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=CONFIG["use_amp"]):
                    logits_val = model(x_val)
                    loss_val = criterion(logits_val.view(-1, vocab_size), y_val.view(-1))
                val_loss += loss_val.item()
                val_pbar.set_postfix({'Val Loss': f"{loss_val.item():.4f}"})
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f}s")

        train_loss_per_epoch.append(train_loss_epoch)
        val_loss_per_epoch.append(val_loss)
        if have_tensorboard:
            tb_writer.add_scalar('Loss/epoch_val', val_loss, global_step)

        # Save the model at the end of the epoch
        epoch_model_path = os.path.join(CONFIG["checkpoint_dir"], f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        logger.info(f"Model saved for epoch {epoch} at: {epoch_model_path}")

    final_pth_path = os.path.join(CONFIG["checkpoint_dir"], "rpt1.pth")
    final_pt_path = os.path.join(CONFIG["checkpoint_dir"], "rpt1.pt")
    torch.save(model.state_dict(), final_pth_path)
    torch.save(model, final_pt_path)
    logger.info(f"Final model saved as: {final_pth_path} and {final_pt_path}")

    epochs_range = range(1, CONFIG["num_epochs"] + 1)
    plt.plot(epochs_range, train_loss_per_epoch, label='Train Loss')
    plt.plot(epochs_range, val_loss_per_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(CONFIG["logs_dir"], "losses.png"))
    plt.close()
    logger.info("Losses PNG file saved in logs folder.")

    csv_path = os.path.join(CONFIG["logs_dir"], "losses_per_epoch.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i in range(CONFIG["num_epochs"]):
            f.write(f"{i+1},{train_loss_per_epoch[i]:.4f},{val_loss_per_epoch[i]:.4f}\n")
    logger.info("CSV file with losses per epoch saved in logs folder.")

    txt_path = os.path.join(CONFIG["logs_dir"], "final_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Final Train Loss: {train_loss_per_epoch[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {val_loss_per_epoch[-1]:.4f}\n")
        f.write("Training complete.\n")
    logger.info("TXT file with final stats saved in logs folder.")

    if 'tb_writer' in locals():
        tb_writer.close()

    logger.info("Training complete!")

if __name__ == "__main__":
    train()
