# Copyright (c) 2025 by Robert Senatorov. All rights reserved.
# tokenize_data.py
"""
Reddit Q&A Tokenizer & SentencePiece Trainer
---------------------------------------------
This script processes saved Q&A pairs (with chain-of-thought) from a JSON file
and tokenizes them using SentencePiece with a vocabulary of 16,000 tokens.
The vocabulary consists of:
  1) Special tokens: <pad>, <eos>, <usr>, <cot>, <bot>
  2) The default <unk> token from SentencePiece
  3) Additional tokens learned from the dataset

Each record in the input JSON is expected to have the fields:
  - "question"
  - "chain_of_thought"
  - "answer"

Tokenization:
  The tokens are combined into a single sequence in the order:
    <usr> question tokens <cot> chain-of-thought tokens <bot> answer tokens <eos>

  Sequences are limited to a fixed length of 128 tokens; shorter sequences are padded 
  with <pad> and longer ones are skipped.

Output Files:
  - out_text.jsonl   : Each line is a JSON object {"text": "..."} with space-delimited tokens.
  - out_tokens.jsonl : Each line is a JSON object {"tokens": [...]} with token IDs.
  - vocab/tokenizer.model : SentencePiece model file
  - vocab/tokenizer.vocab : SentencePiece vocabulary file
"""

#######################
#      Imports        #
#######################
import os
import json
import logging
import time
import sentencepiece as spm
import concurrent.futures
from tqdm import tqdm

#######################
#  Global Directories #
#######################
DATASET_DIR = "dataset"
VOCAB_DIR = os.path.join(DATASET_DIR, "vocab")
INPUT_JSON = os.path.join(DATASET_DIR, "pairs.json")
OUTPUT_TEXT_JSONL = os.path.join(DATASET_DIR, "out_text.jsonl")
OUTPUT_TOKENS_JSONL = os.path.join(DATASET_DIR, "out_tokens.jsonl")
SP_MODEL_PATH = os.path.join(VOCAB_DIR, "tokenizer.model")
TEMP_CORPUS_PATH = os.path.join(DATASET_DIR, "temp_corpus.txt")

#######################
#  Tokenizer Settings #
#######################
MAX_SEQ_LEN = 128
VOCAB_SIZE = 16000
NUM_THREADS = 8

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

#######################
#   Logging Setup     #
#######################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###############################
#   SentencePiece Tokenizer   #
###############################
class SPTokenizer:
    def __init__(self):
        self.sp = None
        self.special_tokens = SPECIAL_TOKENS
        self.unk_token = UNK_TOKEN
        self.pad_token = PAD_TOKEN
        self.eos_token = EOS_TOKEN

    def extract_raw_text(self, data, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in tqdm(data, desc="Extracting text", unit="record"):
                for field in ["question", "chain_of_thought", "answer"]:
                    text = record.get(field, "").strip()
                    if text:
                        f.write(text + "\n")
        return output_path

    def train(self, input_file, model_prefix, vocab_size):
        start_time = time.time()
        logger.info("Training SentencePiece model...")

        adjusted_vocab_size = vocab_size - len(self.special_tokens)

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=adjusted_vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            user_defined_symbols=self.special_tokens,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            hard_vocab_limit=False,
            normalization_rule_name="identity"
        )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_prefix}.model")

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Final SentencePiece vocabulary size: {self.sp.get_piece_size()}")

        return f"{model_prefix}.model"

    def tokenize(self, text):
        for token in self.special_tokens:
            safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
            text = text.replace(token, safe_token)
        return self.sp.encode_as_pieces(text)

    def tokens_to_ids(self, tokens):
        output_ids = []
        for t in tokens:
            tid = self.sp.piece_to_id(t)
            if tid == -1:
                tid = self.sp.unk_id()
            output_ids.append(tid)
        return output_ids

    def load_model(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

###################################
#     Record Processing Function  #
###################################
def process_record(args):
    record, tokenizer = args
    question = record.get("question", "").strip()
    cot = record.get("chain_of_thought", "").strip()
    answer = record.get("answer", "").strip()

    if not question or not cot or not answer:
        return False, None, "empty_field"

    q_tokens = tokenizer.tokenize(question)
    c_tokens = tokenizer.tokenize(cot)
    a_tokens = tokenizer.tokenize(answer)

    combined_tokens = [USR_TOKEN] + q_tokens + [COT_TOKEN] + c_tokens + [BOT_TOKEN] + a_tokens + [EOS_TOKEN]

    if len(combined_tokens) > MAX_SEQ_LEN:
        return False, None, "too_long"

    if len(combined_tokens) < MAX_SEQ_LEN:
        combined_tokens += [PAD_TOKEN] * (MAX_SEQ_LEN - len(combined_tokens))

    token_ids = tokenizer.tokens_to_ids(combined_tokens)
    return True, (combined_tokens, token_ids), None

#######################
#        Main         #
#######################
def main():
    start_total_time = time.time()
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)

    if not os.path.exists(INPUT_JSON):
        logger.error(f"Input file not found: {INPUT_JSON}")
        return

    loading_start = time.time()
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {INPUT_JSON}: {e}")
            return
    loading_time = time.time() - loading_start
    logger.info(f"Loaded {len(records)} records from {INPUT_JSON} in {loading_time:.2f} seconds")

    tokenizer = SPTokenizer()
    tokenizer.extract_raw_text(records, TEMP_CORPUS_PATH)
    model_prefix = os.path.join(VOCAB_DIR, "tokenizer")
    model_path = tokenizer.train(TEMP_CORPUS_PATH, model_prefix, VOCAB_SIZE)

    logger.info(f"Processing {len(records)} records using {NUM_THREADS} threads...")
    processing_start = time.time()

    text_lines = []
    token_lines = []
    total_count = len(records)
    success_count = 0
    skipped_empty = 0
    skipped_too_long = 0

    batch_size = 1000
    with tqdm(total=total_count, desc="Processing records", unit="record") as pbar:
        for i in range(0, total_count, batch_size):
            batch = records[i:min(i+batch_size, total_count)]
            args_list = [(record, tokenizer) for record in batch]

            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                results = list(executor.map(process_record, args_list))

            for success, result, skip_reason in results:
                if success:
                    tokens, token_ids = result
                    text_lines.append({"text": " ".join(tokens)})
                    token_lines.append({"tokens": token_ids})
                    success_count += 1
                else:
                    if skip_reason == "empty_field":
                        skipped_empty += 1
                    elif skip_reason == "too_long":
                        skipped_too_long += 1

            pbar.update(len(batch))

    processing_time = time.time() - processing_start

    #######################
    #   Writing Outputs   #
    #######################
    logger.info("Writing output files...")
    writing_start = time.time()

    with open(OUTPUT_TEXT_JSONL, "w", encoding="utf-8") as f_txt:
        for item in text_lines:
            f_txt.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUTPUT_TOKENS_JSONL, "w", encoding="utf-8") as f_tok:
        for item in token_lines:
            f_tok.write(json.dumps(item, ensure_ascii=False) + "\n")

    writing_time = time.time() - writing_start

    if os.path.exists(TEMP_CORPUS_PATH):
        os.remove(TEMP_CORPUS_PATH)

    #######################
    #   Final Logging     #
    #######################
    total_time = time.time() - start_total_time
    logger.info("Processing complete.")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Loading time: {loading_time:.2f} seconds")
    logger.info(f"Processing time: {processing_time:.2f} seconds ({processing_time/total_count:.4f} seconds per record)")
    logger.info(f"Writing time: {writing_time:.2f} seconds")
    logger.info(f"Total records processed: {total_count}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Skipped (empty fields): {skipped_empty}")
    logger.info(f"Skipped (too long): {skipped_too_long}")
    logger.info("Created files:")
    logger.info(f"  {OUTPUT_TEXT_JSONL}")
    logger.info(f"  {OUTPUT_TOKENS_JSONL}")
    logger.info(f"  {SP_MODEL_PATH}")
    logger.info(f"  {model_prefix}.vocab")

if __name__ == "__main__":
    main()
