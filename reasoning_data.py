# Copyright (c) 2025 by Robert Senatorov. All rights reserved.
# reasoning_data.py
"""
Reddit Q&A Chain-of-Thought Generator
--------------------------------------
This script uses Ollama to generate chain-of-thought explanations for Reddit
question/answer pairs. It processes input CSV files and outputs a JSON file
with normalized text fields.
"""

#######################
#      Imports        #
#######################
import csv
import subprocess
import concurrent.futures
import os
import sys
import json
import time
import logging
import signal
from typing import Dict, List, Tuple
from tqdm import tqdm

#######################
# Global Directories  #
#######################
LOG_DIR = "logs"
DATASET_DIR = "dataset"
CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# Ensure required directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

#######################
#   Logging Setup     #
#######################
LOG_FILE = os.path.join(LOG_DIR, "ollama_processor.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OllamaProcessor")

###############################
# Helper Functions & Settings #
###############################
def set_csv_field_size_limit():
    """
    Set the CSV field size limit safely, reducing the limit if necessary 
    to avoid OverflowError on Windows.
    """
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

set_csv_field_size_limit()

def normalize_text(text: str) -> str:
    """
    Normalize text by replacing newline characters with literal "\n" so that
    the output JSON file contains one-line-per-record strings.
    """
    return text.replace("\n", "\\n").replace("\r", "")

###############################
#      Default Configuration  #
###############################
DEFAULT_CONFIG = {
    "dataset_dir": DATASET_DIR,
    "questions_file": "reddit_questions.csv",
    "answers_file": "reddit_answers_long.csv",
    "output_file": "pairs.json",     # JSON output
    "model_name": "llama3.2:1b",     # Updated model to use Llama 3.2 1B (non-quantized)
    "max_workers": 10,               # Run 10 Q&A pairs concurrently
    "batch_size": 50,
    "delimiter": ";",                # Only used for reading input CSVs.
    "timeout": 60,
    "retry_attempts": 3,
    "retry_delay": 2,
    "prompt_template": (
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Now, as a free-thinking persona named \"fredda\", generate exactly one line of text—a chain-of-thought explanation showing how you derived the answer from the question. Your output MUST be a single line with no newline characters whatsoever and must be under 128 characters. You are REQUIRED to include the tokens \"hmm\", \"aha\", and \"ah ha\" (all in lowercase) at least once each; failure to include any of these tokens renders your response invalid. Do not repeat the question or answer or include extra commentary—output only the chain-of-thought reasoning."
    )
}

###############################
#     Custom Exceptions       #
###############################
class OllamaError(Exception):
    """Base exception for Ollama-related errors"""
    pass

class OllamaTimeoutError(OllamaError):
    """Exception raised when an Ollama request times out"""
    pass

class OllamaConnectionError(OllamaError):
    """Exception raised when connection to Ollama fails"""
    pass

###############################
#      Data Processor Class   #
###############################
class DataProcessor:
    """
    Handles the loading, processing, and generation of chain-of-thought 
    for Reddit Q&A pairs.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.questions_file = os.path.join(self.config["dataset_dir"], self.config["questions_file"])
        self.answers_file = os.path.join(self.config["dataset_dir"], self.config["answers_file"])
        self.output_file = os.path.join(self.config["dataset_dir"], self.config["output_file"])
        self._ensure_directories()
        self.processed_count = 0
        self.error_count = 0
        # Setup graceful shutdown signals
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
    def _ensure_directories(self):
        """Ensure that the dataset directory exists."""
        os.makedirs(self.config["dataset_dir"], exist_ok=True)
    
    def _handle_interrupt(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.warning(f"Received interrupt signal. Processed {self.processed_count} items with {self.error_count} errors. Shutting down...")
        sys.exit(0)
        
    def load_data(self) -> Tuple[List[Dict], Dict]:
        """
        Load question and answer data from CSV files.
        
        Returns:
            A tuple (answers, questions_lookup) where questions_lookup is a dict keyed by question ID.
        """
        try:
            logger.info(f"Loading questions from {self.questions_file}")
            with open(self.questions_file, newline='', encoding='utf-8', errors='replace') as fq:
                reader_q = csv.DictReader(fq, delimiter=self.config["delimiter"])
                questions = list(reader_q)
            
            logger.info(f"Loading answers from {self.answers_file}")
            with open(self.answers_file, newline='', encoding='utf-8', errors='replace') as fa:
                reader_a = csv.DictReader(fa, delimiter=self.config["delimiter"])
                answers = list(reader_a)
                
            questions_lookup = {row['id']: row for row in questions}
            logger.info(f"Loaded {len(questions)} questions and {len(answers)} answers")
            return answers, questions_lookup
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_tasks(self, answers: List[Dict], questions_lookup: Dict) -> List[Tuple[Dict, Dict]]:
        """
        Prepare a list of tasks pairing each question with its top 3 most upvoted answers.
        If a question has no corresponding answers, it is skipped.
        """
        tasks_by_question = {}
        # Group answers by question id (q_id)
        for answer in answers:
            q_id = answer.get('q_id', '').strip()
            if q_id in questions_lookup:
                tasks_by_question.setdefault(q_id, []).append(answer)

        tasks = []
        # For each question, sort the answers by vote count (descending) and take the top 3
        for q_id, answer_list in tasks_by_question.items():
            sorted_answers = sorted(answer_list, key=lambda a: float(a.get('votes', 0)), reverse=True)
            top_answers = sorted_answers[:3]
            for answer in top_answers:
                tasks.append((questions_lookup[q_id], answer))
        
        logger.info(f"Prepared {len(tasks)} tasks for processing")
        return tasks
    
    def run_ollama(self, prompt: str) -> str:
        """
        Invoke the Ollama model with a given prompt, with retry logic.
        """
        for attempt in range(self.config["retry_attempts"]):
            try:
                result = subprocess.run(
                    ["ollama", "run", self.config["model_name"], prompt],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                    timeout=self.config["timeout"]
                )
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                logger.warning(f"Ollama timed out (attempt {attempt+1}/{self.config['retry_attempts']})")
                if attempt == self.config["retry_attempts"] - 1:
                    raise OllamaTimeoutError("Ollama request timed out after multiple attempts")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Ollama error: {e.stderr} (attempt {attempt+1}/{self.config['retry_attempts']})")
                if attempt == self.config["retry_attempts"] - 1:
                    if "connection refused" in str(e).lower():
                        raise OllamaConnectionError("Failed to connect to Ollama. Is the service running?")
                    raise OllamaError(f"Ollama process error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error: {e} (attempt {attempt+1}/{self.config['retry_attempts']})")
                if attempt == self.config["retry_attempts"] - 1:
                    raise
            time.sleep(self.config["retry_delay"])
    
    def process_pair(self, question_row: Dict, answer_row: Dict) -> Dict:
        """
        Process a single question-answer pair: normalize text, construct prompt, 
        run the model, and return the result.
        """
        try:
            question_text = normalize_text(question_row.get('text', '').strip())
            answer_text = normalize_text(answer_row.get('text', '').strip())
            
            if not question_text or not answer_text:
                raise ValueError("Empty question or answer text")
                
            prompt = self.config["prompt_template"].format(question=question_text, answer=answer_text)
            chain_of_thought = self.run_ollama(prompt)
            chain_of_thought = normalize_text(chain_of_thought)
            
            return {
                'question': question_text,
                'chain_of_thought': chain_of_thought,
                'answer': answer_text
            }
        except OllamaError as e:
            self.error_count += 1
            return {
                'question': normalize_text(question_row.get('text', '')),
                'chain_of_thought': f"Error: {e}",
                'answer': normalize_text(answer_row.get('text', ''))
            }
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing pair: {e}")
            return {
                'question': normalize_text(question_row.get('text', '')),
                'chain_of_thought': f"Processing error: {e}",
                'answer': normalize_text(answer_row.get('text', ''))
            }
    
    def process_batch(self, tasks: List[Tuple[Dict, Dict]]) -> List[Dict]:
        """
        Process a batch of tasks in parallel and return a list of results.
        """
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            futures = {
                executor.submit(self.process_pair, q_row, a_row): (q_row, a_row)
                for (q_row, a_row) in tasks
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    row = future.result()
                    results.append(row)
                    self.processed_count += 1
                except Exception as ex:
                    self.error_count += 1
                    logger.error(f"Failed to process task: {ex}")
        return results
    
    def process_all(self, tasks: List[Tuple[Dict, Dict]]):
        """
        Process all tasks in batches and write the results to a JSON file.
        The output JSON file will contain a list of records.
        """
        first_record = True
        try:
            with open(self.output_file, 'w', encoding='utf-8') as fout:
                fout.write("[\n")
                total_batches = (len(tasks) + self.config["batch_size"] - 1) // self.config["batch_size"]
                for i in range(0, len(tasks), self.config["batch_size"]):
                    batch = tasks[i:i+self.config["batch_size"]]
                    logger.info(f"Processing batch {i//self.config['batch_size'] + 1}/{total_batches}")
                    records = self.process_batch(batch)
                    for record in records:
                        if not first_record:
                            fout.write(",\n")
                        else:
                            first_record = False
                        fout.write(json.dumps(record, ensure_ascii=False))
                    fout.flush()
                fout.write("\n]\n")
        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
            raise

###############################
#      Utility Functions      #
###############################
def check_ollama_installation() -> bool:
    """
    Check if Ollama is installed and available in the system PATH.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_model_availability(model_name: str) -> bool:
    """
    Check if the specified model is available in Ollama.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
        return model_name in result.stdout
    except Exception:
        return False

def load_config(config_path: str = CONFIG_FILE) -> Dict:
    """
    Load configuration from the specified JSON file, or create one with defaults if not found.
    """
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                for key, value in user_config.items():
                    if key in config:
                        config[key] = value
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config file: {e}. Using default configuration.")
    else:
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Created default configuration file at {config_path}")
    return config

###############################
#         Main Entry          #
###############################
def main():
    """
    Main entry point: load configuration, check dependencies, load data, prepare tasks,
    and process the chain-of-thought generation, writing results to a JSON file.
    """
    config = load_config()
    
    if not check_ollama_installation():
        logger.error("Ollama is not installed or not available in PATH. Please install Ollama first.")
        print("\nOllama Installation Instructions:")
        print("1. Visit https://ollama.com/download to download and install Ollama")
        print("2. After installation, start Ollama service (e.g., run 'ollama serve')")
        print("3. Run 'ollama pull llama3.2:1b' to download the required model")
        sys.exit(1)
    
    if not check_model_availability(config["model_name"]):
        logger.warning(f"Model '{config['model_name']}' not found in Ollama.")
        print(f"\nTo pull the required model, run:")
        print(f"ollama pull {config['model_name']}")
        response = input("Would you like to pull the model now? (y/n): ")
        if response.lower() == 'y':
            print(f"Pulling model {config['model_name']}...")
            subprocess.run(["ollama", "pull", config["model_name"]], check=True,
                           encoding="utf-8", errors="replace")
        else:
            sys.exit(1)
    
    try:
        print("\nRunning with the following configuration:")
        print(f"- Model: {config['model_name']}")
        print(f"- Workers: {config['max_workers']}")
        print(f"- Batch size: {config['batch_size']}")
        print(f"- Dataset directory: {config['dataset_dir']}")
        print(f"- Output: {os.path.join(config['dataset_dir'], config['output_file'])}")
        print("\nTo change these settings, edit the config file in the 'config' folder and restart the program.\n")
        
        processor = DataProcessor(config)
        answers, questions_lookup = processor.load_data()
        tasks = processor.prepare_tasks(answers, questions_lookup)
        
        start_time = time.time()
        processor.process_all(tasks)
        elapsed_time = time.time() - start_time
        
        logger.info("Processing complete!")
        logger.info(f"Processed {processor.processed_count} pairs in {elapsed_time:.2f} seconds")
        logger.info(f"Encountered {processor.error_count} errors")
        logger.info(f"Results saved to {processor.output_file}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
