# BERT Depression Classification

This repository contains the implementation of a BERT-based model for detecting depression severity levels from text data, specifically Reddit posts. The project is designed for research purposes, as described in our paper submitted to AIHW 2025 IEEE (Hyderabad). The model leverages BERT (Bidirectional Encoder Representations from Transformers) with a custom focal loss function to handle class imbalance, text augmentation for minority classes, and SHAP (SHapley Additive exPlanations) for model interpretability.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Results](#results)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Project Overview
This project implements a BERT-based classifier to predict depression severity levels (e.g., none, mild, moderate, severe) from Reddit posts. The pipeline includes:
- **Data Preprocessing**: Cleaning text by removing URLs, mentions, hashtags, and converting emojis to text.
- **Text Augmentation**: Using `nlpaug` to augment minority classes to address class imbalance.
- **Model Training**: Fine-tuning a BERT model (`bert-base-uncased`) with a custom focal loss to handle imbalanced classes.
- **Evaluation**: Metrics including accuracy, precision, recall, F1-score, AUC-ROC, and precision-recall AUC.
- **Interpretability**: SHAP explanations to identify influential tokens in predictions.
- **High-Risk Alerts**: Flagging high-confidence severe cases for potential intervention.

The code is implemented in Python using PyTorch, Hugging Face Transformers, and SHAP for interpretability.

## Dataset
The dataset used is `Reddit_depression_dataset.csv`, assumed to be collected with user consent under Reddit's terms of service or a research-approved process. It contains text posts and corresponding depression severity labels. Due to privacy and ethical considerations, the dataset is not included in this repository. Users must provide their own dataset in the format:
- Columns: `text` (Reddit post content), `label` (depression severity: none, mild, moderate, severe).

Place the dataset in the `data/` directory before running the code.

## Features
- **Text Cleaning**: Removes URLs, mentions, hashtags, and converts emojis to text representations.
- **Data Augmentation**: Synonym-based augmentation for minority classes using WordNet.
- **Focal Loss**: Custom loss function to address class imbalance in depression severity classification.
- **Model Training**: Fine-tuned BERT model with balanced class weights and gradient accumulation.
- **Evaluation Metrics**: Comprehensive metrics including accuracy, precision, recall, F1-score (weighted and macro), AUC-ROC, and precision-recall AUC.
- **SHAP Interpretability**: Token-level and probability-based SHAP analysis for model transparency.
- **High-Risk Detection**: Alerts for high-confidence severe depression cases (confidence > 0.9).
- **Visualization**: Confusion matrix and SHAP summary plots for model evaluation and interpretability.

## Requirements
The following Python packages are required:
```plaintext
pandas
numpy
torch
transformers
transformers-interpret
scikit-learn
nlpaug
nltk
shap
matplotlib
seaborn
joblib
emoji
```

Optional (for GPU acceleration):
- CUDA-compatible GPU
- PyTorch with CUDA support

## Installation
1. Clone this repository:
   ```bash
   https://github.com/TechieTripathi/bert-depression-classification.git
   cd bert-depression-classification
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK resources:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('averaged_perceptron_tagger_eng')
   ```
5. Place your dataset (`Reddit_depression_dataset.csv`) in the `data/` directory.

To generate a `requirements.txt` file, run:
```bash
pip freeze > requirements.txt
```

## Usage
1. Ensure the dataset is in the `data/` directory.
2. Run the main script:
   ```bash
   python bert_depression_classification.py
   ```
3. The script will:
   - Preprocess and augment the data.
   - Train the BERT model and save it to `bert_model/`.
   - Generate predictions and save them to `.npy` files.
   - Compute evaluation metrics and log them to `depression_detection.log`.
   - Generate a confusion matrix (`bert_confusion_matrix.png`).
   - Produce SHAP visualizations (`shap_bert_plot_*.html`, `shap_bert_annotated_*.png`, `shap_token_summary.png`, `shap_bert_summary.png`).
   - Log alerts for high-risk cases.

To load the trained model for inference:
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert_model')
tokenizer = BertTokenizer.from_pretrained('bert_model')
model.eval()

# Example inference
text = ["I feel hopeless and tired all the time."]
encodings = tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**encodings)
    probs = torch.softmax(outputs.logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
print(probs, preds)
```

## Directory Structure
```
bert-depression-classification/
├── bert_confusion_matrix.png      # Confusion matrix plot
├── bert_depression_classification.py  # Main script
├── bert_model/                    # Trained BERT model and tokenizer
├── bert_preds_test.npy            # Test set predictions
├── bert_preds_train.npy           # Training set predictions
├── bert_probs_test.npy            # Test set probabilities
├── bert_probs_train.npy           # Training set probabilities
├── data/                          # Dataset directory (user-provided)
│   └── Reddit_depression_dataset.csv
├── depression_detection.log       # Log file with metrics and alerts
├── helper/                        # Supporting documentation and diagrams
├── label_encoder.pkl              # Label encoder for depression severity
├── results/                       # Training checkpoints
├── shap_bert_plot_*.html          # SHAP visualization HTML files
├── shap_bert_annotated_*.png      # Annotated SHAP plots
├── shap_bert_summary.png          # SHAP summary for BERT probabilities
├── shap_token_summary.png         # SHAP summary for important tokens
├── test_data.csv                  # Test dataset split
├── train_data.csv                 # Training dataset split
└── README.md                      # This file
```

## Results
The model achieves the following performance (example metrics, replace with your actual results):
- **Accuracy**: ~0.85
- **Precision (weighted)**: ~0.84
- **Recall (weighted)**: ~0.85
- **F1-score (weighted)**: ~0.84
- **F1-score (macro)**: ~0.80
- **AUC-ROC (ovr)**: ~0.90
- **Precision-Recall AUC (class 'minimum')**: ~0.88

Check `depression_detection.log` for detailed metrics and `bert_confusion_matrix.png` for the confusion matrix. SHAP visualizations provide insights into token-level contributions to predictions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## Citation
If you use this code in your research, please cite our paper:
```
[Your Name(s)]. (2025). "BERT-Based Depression Severity Classification from Social Media Text." In *Proceedings of AIHW 2025 IEEE, Hyderabad*.
``` -->

## Contact
For questions or issues, please open a GitHub issue or contact [vishnutripathi.ai@example.com].