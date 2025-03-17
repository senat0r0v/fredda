# Copyright (c) 2025 by Robert Senatorov. All rights reserved.
# inference.py
"""
Updated Chat Inference Script with Chain-of-Thought Reasoning
-------------------------------------------------------------
This script performs inference using a Transformer-based chatbot model,
generating both the model's internal chain-of-thought and its final answer.
It uses the SentencePiece tokenizer produced during training (tokenizer.model and tokenizer.vocab)
to tokenize the input and properly detokenize the generated output.

Training sequences were constructed as:
    <usr> question tokens <cot> chain-of-thought tokens <bot> answer tokens <eos> [<pad>...]

For inference, the prompt is constructed exactly as:
    <usr> question tokens <cot>
This signals that the user has finished and that the model should now generate its
chain-of-thought (tokens between <cot> and <bot>) and its final answer (tokens between <bot> and <eos>).

Usage:
  1) Ensure training is completed and the following files exist:
         - checkpoints/rpt1.pth         (or model_epoch_X.pth)
         - dataset/vocab/tokenizer.model
  2) Run: python inference.py
  3) Enter a question at the "User >>" prompt. Type "quit" to exit.
"""

#######################
#      Imports        #
#######################
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

###############################
#   Logging Setup             #
###############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

###############################
#   Configuration for Inference
###############################
CONFIG = {
    "checkpoint_dir": "checkpoints",
    "final_model": "rpt1.pth",            # or "model_epoch_X.pth"
    "sp_model": "dataset/vocab/tokenizer.model",
    "max_seq_len": 128,                   # Must match training sequence length
    "max_new_tokens": 128,                # Maximum new tokens to generate
    "temperature": 0.8,                   # Sampling temperature (1.0 = no scaling)
    "top_k": 10,                          # Top-k sampling parameter
    "top_p": 0.99,                        # Nucleus (top-p) sampling parameter
    "repetition_penalty": 1.5,            # Penalty for repeated tokens
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

###############################
#   Special Tokens            #
###############################
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
USR_TOKEN = "<usr>"
COT_TOKEN = "<cot>"
BOT_TOKEN = "<bot>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    EOS_TOKEN,
    USR_TOKEN,
    COT_TOKEN,
    BOT_TOKEN
]

###############################
#   Tokenization Helpers      #
###############################
def create_prompt(user_text, sp):
    """
    Constructs the prompt exactly as in training:
      <usr> [user_text tokens] <cot>
    This signals that the user is finished and that the model should now generate its chain-of-thought
    followed by its answer.
    
    Handles special tokens the same way as in training script.
    """
    user_text = user_text.strip()
    
    # Protect special tokens exactly as done in training script
    processed_text = user_text
    for token in SPECIAL_TOKENS:
        safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
        processed_text = processed_text.replace(token, safe_token)
    
    pieces = sp.encode_as_pieces(processed_text)
    
    # Return the proper sequence with special tokens directly
    return [USR_TOKEN] + pieces + [COT_TOKEN]

###############################
#   Model Definition          #
###############################
class TransformerDecoder(nn.Module):
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
        hidden_states = self.token_emb(x) + self.pos_emb(positions)
        hidden_states = self.dropout(hidden_states)

        # Create a causal mask for autoregressive generation.
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        memory = torch.zeros_like(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, memory, tgt_mask=causal_mask)
        hidden_states = self.ln_f(hidden_states)
        logits = self.fc_out(hidden_states)
        return logits

###############################
#   Sampling Utilities        #
###############################
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    if penalty <= 1.0:
        return logits
    for token_id in set(generated_ids):
        logits[token_id] /= penalty
    return logits

def top_k_top_p_filtering(probs, top_k=None, top_p=None):
    if top_k is None and top_p is None:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_idx = probs.size(-1)
    if top_k is not None and top_k > 0 and top_k < probs.size(-1):
        cutoff_idx = min(cutoff_idx, top_k)
    if top_p is not None and top_p < 1.0:
        mask = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
        if len(mask) > 0:
            cutoff_idx = min(cutoff_idx, mask[0].item() + 1)

    new_probs = torch.zeros_like(probs)
    new_probs[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
    return new_probs

@torch.no_grad()
def generate_text(
    model,
    prompt_token_ids,
    max_seq_len,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    eos_id,
    device
):
    """
    Generates token IDs autoregressively given the prompt.
    """
    model.eval()
    input_ids = list(prompt_token_ids)
    generated_ids = []
    for _ in range(max_new_tokens):
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[-max_seq_len:]
        inp_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(inp_tensor)  # Shape: [1, T, vocab_size]
        last_logits = logits[0, -1, :]
        if repetition_penalty > 1.0:
            last_logits = apply_repetition_penalty(last_logits, generated_ids, penalty=repetition_penalty)
        last_logits = last_logits / max(temperature, 1e-8)
        probs = torch.softmax(last_logits, dim=-1)
        probs = top_k_top_p_filtering(probs, top_k=top_k, top_p=top_p)
        probs = probs / probs.sum()
        next_id = int(torch.multinomial(probs, 1))
        generated_ids.append(next_id)
        input_ids.append(next_id)
        if next_id == eos_id:
            break
    return generated_ids

###############################
#   Main Inference Loop       #
###############################
def main():
    device = torch.device(CONFIG["device"])
    logger.info(f"Using device: {device}")

    # Load SentencePiece model for tokenization and detokenization.
    sp = spm.SentencePieceProcessor()
    sp.load(CONFIG["sp_model"])
    logger.info(f"Loaded SentencePiece model from {CONFIG['sp_model']}")

    vocab_size = sp.get_piece_size()

    # Retrieve special token IDs from SentencePiece.
    try:
        eos_id = sp.piece_to_id(EOS_TOKEN)
        pad_id = sp.piece_to_id(PAD_TOKEN)
        bot_id = sp.piece_to_id(BOT_TOKEN)
        cot_id = sp.piece_to_id(COT_TOKEN)
    except Exception as e:
        logger.error(f"Error retrieving special token IDs: {e}")
        sys.exit(1)

    # Build the Transformer model.
    model = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        max_seq_len=CONFIG["max_seq_len"]
    ).to(device)

    # Load the trained checkpoint.
    model_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["final_model"])
    if not os.path.isfile(model_path):
        logger.error(f"Model file '{model_path}' not found!")
        sys.exit(1)

    raw_state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    logger.info(f"Loaded model from: {model_path}")

    print("Ready for inference! Type 'quit' to exit.\n")

    while True:
        user_input = input("User >> ").strip()
        if user_input.lower() == "quit":
            break

        # Construct prompt: <usr> question tokens <cot>
        prompt_pieces = create_prompt(user_input, sp)
        prompt_token_ids = [sp.piece_to_id(piece) for piece in prompt_pieces]

        # Generate tokens from the model.
        generated_ids = generate_text(
            model=model,
            prompt_token_ids=prompt_token_ids,
            max_seq_len=CONFIG["max_seq_len"],
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
            top_p=CONFIG["top_p"],
            repetition_penalty=CONFIG["repetition_penalty"],
            eos_id=eos_id,
            device=device
        )

        # Convert generated IDs to SentencePiece pieces.
        generated_pieces = [sp.id_to_piece(token_id) for token_id in generated_ids]

        # Find where the chain-of-thought ends and the answer begins
        try:
            bot_index = generated_pieces.index(BOT_TOKEN)
            # Chain-of-thought tokens are between the prompt (which ends with <cot>) and <bot>
            cot_tokens = generated_pieces[:bot_index]
            # Answer tokens are after <bot> and before <eos> if present
            answer_tokens = generated_pieces[bot_index + 1:]
        except ValueError:
            # If <bot> token not found, assume all output is chain-of-thought
            cot_tokens = generated_pieces
            answer_tokens = []

        # Remove the EOS token from the answer if present.
        if EOS_TOKEN in answer_tokens:
            eos_index = answer_tokens.index(EOS_TOKEN)
            answer_tokens = answer_tokens[:eos_index]

        # Detokenize using SentencePiece's decode_pieces.
        cot_text = sp.decode_pieces(cot_tokens)
        answer_text = sp.decode_pieces(answer_tokens)

        print("\nChain-of-Thought:", cot_text)
        print("Answer:", answer_text)
        print("-" * 50)

if __name__ == "__main__":
    main()