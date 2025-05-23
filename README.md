# Fredda: Functional Response Emulation Device Dialogue Analysis

Welcome to Fredda, a Redditor chain-of-thought bot that generates both answers and its internal reasoning. Fredda is inspired by the classic Fred chatbot and uses a transformer-based model for clear, transparent dialogue processing.

![example](https://github.com/user-attachments/assets/98568c71-e8c8-4c84-94b4-8421f68e9b0f)

## Getting Started
1. Clone this repository.
2. Create a new Conda environment (Python 3.9 is recommended):
   ```
   conda create -n fredda python=3.9
   ```
   ```
   conda activate fredda
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
6. Make sure the following files are in place:
   - **Dataset Artifacts:** Download **out_tokens.jsonl** from [this Google Drive folder](https://drive.google.com/drive/folders/1H2kSB35jWxHZV03WUWr9rLtgv2uOB21A?usp=sharing) and place it in the **dataset** folder. You must do this if you want to train the model yourself with the same or different parameters.
   - **Vocab Folder:** The vocab folder (containing tokenizer.model and tokenizer.vocab) is included with the repository clone.
   - **Model Checkpoint:** Download **rpt1.pth** (the model is above 100MB) from [this Google Drive folder](https://drive.google.com/drive/folders/1jN9fzH9SA4D3Ac8-rNh2HDwoR62eSvVA?usp=sharing) and place it in the **checkpoints** folder ***(you will need to make this folder)***. You must do this if you want to interact with the bot.
7. To run the command-line inference, type:
   ```
   python inference.py
   ```
   To run the Discord bot, type:
   ```
   python discord_inference.py
   ```
   (Remember to update the Discord bot token in the configuration.)

If you would like to generate your own reasoning dataset by using reasoning_data.py please install ollama and download llama 3.2:1b in the terminal with ```ollama run llama3.2:1b```

> Note: Only the necessary dataset files are published:
> - **out_tokens.jsonl** (download from Google Drive)
> - The **vocab** folder (contains tokenizer.model and tokenizer.vocab)
> - The model checkpoint **rpt1.pth** (download from Google Drive)
> All source code is available for full reproducibility.

## Table of Contents
1. Introduction
2. Project Overview
3. Components
   - Model Architecture & Training
   - Inference Pipelines
   - Reasoning Data Generation
   - Tokenization & Dataset Preparation
4. Datasets
5. Training Environment & Logs
6. Published Artifacts
7. Pipeline Diagram
8. Terminal Usage Example
9. Collaborators and Acknowledgments
10. Summary and Future Work
11. License

## Introduction
Fredda is a Redditor chain-of-thought bot that produces both an answer and a hidden reasoning process (chain-of-thought). When you interact with Fredda, it returns a spoiler-wrapped explanation of its internal reasoning followed by the final answer.

## Project Overview
Fredda covers the complete workflow from data preprocessing to real-time inference. Its main parts include:
- **Model Training:** A transformer decoder with 12 layers, 768-dimensional embeddings, and 12 attention heads.
- **Inference:** Available through a command-line interface and a Discord bot.
- **Data Generation:** Scripts generate chain-of-thought explanations for Reddit Q&A pairs.
- **Tokenization:** Uses SentencePiece to ensure consistent tokenization between training and inference.

## Components

### Model Architecture & Training
- Transformer decoder with 12 layers and 768-d embeddings.
- Trained on 341,369 examples and tested on 37,929 examples.
- Sequence length set to 128 (to keep training time reasonable; training took approximately 8 hours using CUDA on an RTX 4070).
- Final model checkpoint saved as "rpt1.pth".
- Final training loss: Train Loss = 4.1279, Validation Loss = 4.1200.
- Detailed logs are stored in the "logs" folder.

### Inference Pipelines
- **CLI Inference:** Run the script with ```python inference.py``` to interact via the terminal.
- **Discord Bot:** Run the script with ```python discord_inference.py```. The bot listens for mentions on specified channels and replies with a spoiler-wrapped chain-of-thought and final answer.

### Reasoning Data Generation
- Script "reasoning_data.py" processes Reddit Q&A pairs.
- Uses Ollama to generate one-line chain-of-thought explanations.
- Output is saved as "pairs.json".

### Tokenization & Dataset Preparation
- Script "tokenize_data.py" trains a SentencePiece tokenizer with a 16,000 token vocabulary.
- Generates two key artifacts:
  - **out_tokens.jsonl** – a tokenized version of the dataset (download from Google Drive if replicating training).
  - The **vocab** folder – containing "tokenizer.model" and "tokenizer.vocab".

## Datasets
Fredda was trained on a reformatted AskReddit Q&A dataset from Kaggle: https://www.kaggle.com/datasets/rodmcn/askreddit-questions-and-answers
- **Training Examples:** 341,369 examples.
- **Testing Examples:** 37,929 examples.
- The data was preprocessed and tokenized to ensure consistent input to the model.

## Training Environment & Logs
- **Environment:** Ubuntu running on Windows Subsystem for Linux (WSL).
- **Hardware:** NVIDIA GeForce RTX 4070 (CUDA 12.8).
- **Mixed Precision:** Enabled (FP16 with TF32) to speed up training.
- **Logs:** Detailed logs are in the "logs" folder:
  - training.log
  - final_stats.txt
  - losses_per_epoch.csv
  - losses.png
  - ollama_processor.log (for reasoning data generation)

## Published Artifacts
Released files include:
- **Dataset Artifacts:**
  - **out_tokens.jsonl** (download from Google Drive and place in the dataset folder)
  - **vocab** folder (contains tokenizer.model and tokenizer.vocab; included with repo clone)
- **Model Checkpoint:**
  - **rpt1.pth** (download from Google Drive and place in the checkpoints folder)
- All source code is available in the repository.

## Pipeline Diagram
A diagram illustrating the full pipeline
![Screenshot 2025-03-17 160946](https://github.com/user-attachments/assets/ccba733a-2370-4626-a757-ae83ac8c9ee2)

## Terminal Usage Example
Below is an example of how the terminal interaction might look:
![Screenshot 2025-03-17 155330](https://github.com/user-attachments/assets/bf7d8965-8c09-4040-add4-bd866e77d86f)

*(This is a sample output; your terminal may show additional details.)*

## Collaborators and Acknowledgments
- **Lead Developer:** Robert Senatorov (sole collaborator).
- **Inspiration:** Inspired by the Fred chatbot.
- **Data Source:** AskReddit Q&A dataset from Kaggle.

## Summary and Future Work
### Achievements:
- Built a transformer-based chatbot that outputs both its answer and internal reasoning.
- Integrated both command-line and Discord-based inference pipelines.
- Made all source code and key dataset artifacts publicly available.

### Future Work:
- Experiment with larger models and fine-tuning parameters.
- Expand integration to additional platforms.
- Enhance data diversity and augmentation strategies.
- Gather and incorporate user feedback to improve performance.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
