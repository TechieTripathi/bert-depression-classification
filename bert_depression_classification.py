import re
import emoji
import pandas as pd
import numpy as np
import torch
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import nltk
import nlpaug.augmenter.word as naw
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers_interpret import SequenceClassificationExplainer

# Note: The dataset (Reddit_depression_dataset.csv) is assumed to be collected with user consent 
# under Reddit's terms of service or a research-approved process. No runtime consent is implemented.

# Set up logging with timestamp and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S IST',
    handlers=[
        logging.FileHandler('depression_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")

# Download required NLTK resources
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    raise

# Set PyTorch CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Focal Loss Implementation
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.5, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# Text Cleaning
def clean_text(text):
    """Clean text by removing URLs, mentions, hashtags, and converting emojis. 
    This step minimizes privacy risks by removing potential indirect identifiers (e.g., URLs, mentions)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#\w+', '', text)
    text = emoji.demojize(text)
    text = text.lower().strip()
    return text

# Text Augmentation for Minority Classes
def augment_text(text, augmenter, num_aug=4):
    try:
        augmented_texts = [augmenter.augment(text)[0] for _ in range(num_aug)]
        return augmented_texts
    except Exception as e:
        logger.warning(f"Augmentation failed for text: {text[:50]}... Error: {e}")
        return [text] * num_aug

# Load and Preprocess Data
try:
    data = pd.read_csv('data/Reddit_depression_dataset.csv', names=['text', 'label'], header=0)
except pd.errors.ParserError as e:
    logger.error(f"ParserError: {e}")
    data = pd.read_csv('data/Reddit_depression_dataset.csv', names=['text', 'label'], on_bad_lines='skip')
    logger.info("Loaded data by skipping bad lines. Check the file for issues.")

data['text'] = data['text'].apply(clean_text)
data = data[data['text'] != ""].dropna().reset_index(drop=True)
logger.info(f"Dataset size after cleaning: {len(data)}")

# Encode Labels
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
joblib.dump(le, 'label_encoder.pkl')
logger.info("LabelEncoder saved to label_encoder.pkl")
label_mapping = dict(zip(range(len(le.classes_)), le.classes_))
logger.info(f"LabelEncoder mapping: {label_mapping}")

# Augment Minority Classes
augmenter = naw.SynonymAug(aug_src='wordnet')
augmented_data = []
for label in data['label'].unique():
    class_data = data[data['label'] == label]
    if len(class_data) < 2000:
        logger.info(f"Augmenting class {le.inverse_transform([label])[0]} with {len(class_data)} samples")
        num_aug = 5 if le.inverse_transform([label])[0] in ['mild', 'severe'] else 4
        for text in class_data['text']:
            augmented_texts = augment_text(text, augmenter, num_aug=num_aug)
            for aug_text in augmented_texts:
                augmented_data.append({'text': aug_text, 'label': label})
augmented_df = pd.DataFrame(augmented_data)
data = pd.concat([data, augmented_df], ignore_index=True)
logger.info(f"Dataset size after augmentation: {len(data)}")

# Analyze Class Imbalance
class_counts = data['label'].value_counts()
total_samples = len(data)
class_percentages = (class_counts / total_samples) * 100
logger.info("Class Distribution:")
for label, count, percentage in zip(le.inverse_transform(class_counts.index), class_counts, class_percentages):
    logger.info(f"Class: {label}, Count: {count}, Percentage: {percentage:.2f}%")
logger.info(f"Total Samples: {total_samples}")

# Compute Class Weights
classes = np.unique(data['label'])
weights = compute_class_weight('balanced', classes=classes, y=data['label'])
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
logger.info(f"Class Weights: {class_weights}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, stratify=data['label'], random_state=42)
train_class_counts = pd.Series(y_train).value_counts()
logger.info("Training Set Class Distribution:")
for label, count in zip(le.inverse_transform(train_class_counts.index), train_class_counts):
    logger.info(f"Class: {label}, Count: {count}")

# Save Train and Test Data
pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('train_data.csv', index=False)
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv('test_data.csv', index=False)
logger.info("Train and test data saved to train_data.csv and test_data.csv")

# BERT Tokenization and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_data(texts, max_length=128):
    return tokenizer(texts.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_data(X_train)
test_encodings = tokenize_data(X_test)

class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = DepressionDataset(train_encodings, y_train.values)
test_dataset = DepressionDataset(test_encodings, y_test.values)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = FocalLoss(gamma=1.5, alpha=class_weights, reduction='mean')
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Compute Metrics Function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Train BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))
model.to(device)
logger.info("BERT model initialized and moved to device")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=7,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    weight_decay=0.01,
    learning_rate=1e-5,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    dataloader_num_workers=4,
    gradient_accumulation_steps=4,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
logger.info("BERT training completed")

# Save BERT Model and Tokenizer
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')
logger.info("BERT model and tokenizer saved to bert_model/")

# Batch Prediction Function
def get_bert_predictions(texts, batch_size=16):
    probs_list = []
    preds_list = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts.tolist(), max_length=128, padding=True, truncation=True, return_tensors='pt')
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        for j in range(len(preds)):
            if np.max(probs[j]) < 0.3 and (probs[j][0] > 0.15 or probs[j][2] > 0.15 or probs[j][3] > 0.15):
                preds[j] = np.argmax(probs[j][[0, 2, 3]])
        probs_list.append(probs)
        preds_list.append(preds)
        torch.cuda.empty_cache()
    return np.vstack(probs_list), np.concatenate(preds_list)

# Generate and Save BERT Predictions
logger.info("Generating BERT predictions on train and test sets...")
bert_probs_train, bert_preds_train = get_bert_predictions(X_train)
bert_probs_test, bert_preds_test = get_bert_predictions(X_test)
np.save('bert_probs_train.npy', bert_probs_train)
np.save('bert_preds_train.npy', bert_preds_train)
np.save('bert_probs_test.npy', bert_probs_test)
np.save('bert_preds_test.npy', bert_preds_test)
logger.info("BERT predictions saved to bert_probs_train.npy, bert_preds_train.npy, bert_probs_test.npy, bert_preds_test.npy")

# Generate Alerts for High-Risk Cases
def generate_alerts(bert_probs_test, bert_preds_test, threshold=0.9):
    high_risk_indices = np.where((bert_probs_test[np.arange(len(bert_preds_test)), bert_preds_test] > threshold) & 
                                 (le.inverse_transform(bert_preds_test) == 'severe'))[0]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
    for idx in high_risk_indices:
        text = X_test.iloc[idx]
        logger.info(f"ALERT [{current_time}]: High-risk case detected (Severe, Confidence: {bert_probs_test[idx, bert_preds_test[idx]]:.4f}): {text[:100]}...")
    return high_risk_indices

logger.info("Generating alerts for high-risk cases...")
high_risk_indices = generate_alerts(bert_probs_test, bert_preds_test)
logger.info(f"Number of high-risk cases detected: {len(high_risk_indices)}")

# Compute Metrics for BERT
logger.info("Computing BERT evaluation metrics...")
accuracy = accuracy_score(y_test, bert_preds_test)
precision_weighted = precision_score(y_test, bert_preds_test, average='weighted')
recall_weighted = recall_score(y_test, bert_preds_test, average='weighted')
f1_weighted = f1_score(y_test, bert_preds_test, average='weighted')
f1_macro = f1_score(y_test, bert_preds_test, average='macro')
roc_auc = roc_auc_score(y_test, bert_probs_test, multi_class='ovr')
precision_pr, recall_pr, _ = precision_recall_curve(y_test, bert_probs_test[:, 1], pos_label=1)
pr_auc = auc(recall_pr, precision_pr)

# Log Metrics
logger.info("BERT Evaluation Metrics:")
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"Precision (weighted): {precision_weighted:.4f}")
logger.info(f"Recall (weighted): {recall_weighted:.4f}")
logger.info(f"F1-score (weighted): {f1_weighted:.4f}")
logger.info(f"F1-score (macro): {f1_macro:.4f}")
logger.info(f"AUC-ROC (ovr): {roc_auc:.4f}")
logger.info(f"Precision-Recall AUC (class 'minimum'): {pr_auc:.4f}")

# Classification Report
logger.info("BERT Classification Report:")
print(classification_report(y_test, bert_preds_test, target_names=le.inverse_transform(np.unique(y_test))))

# Confusion Matrix
logger.info("Generating confusion matrix...")
cm = confusion_matrix(y_test, bert_preds_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.inverse_transform(np.unique(y_test)), yticklabels=le.inverse_transform(np.unique(y_test)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('BERT Confusion Matrix')
plt.savefig('bert_confusion_matrix.png')
plt.close()
logger.info("Confusion matrix saved to bert_confusion_matrix.png")

# SHAP Explanations for BERT
logger.info("Generating SHAP explanations for BERT on sample test texts...")
explainer_bert = SequenceClassificationExplainer(model, tokenizer)
sample_indices = np.random.choice(len(X_test), size=5, replace=False)
for idx in sample_indices:
    try:
        text = X_test.iloc[idx]
        true_label = le.inverse_transform([y_test.iloc[idx]])[0]
        pred_label = le.inverse_transform([bert_preds_test[idx]])[0]
        logger.info(f"\nSample {idx}: True Label: {true_label}, Predicted Label: {pred_label}")
        logger.info(f"Text: {text[:100]}...")
        explanation = explainer_bert(text)
        logger.info(f"SHAP scores for predicted class ({explainer_bert.predicted_class_name}):")
        for word, score in explanation:
            logger.info(f"Word: {word}, SHAP Score: {score:.4f}")
        explainer_bert.visualize(f"shap_bert_plot_{idx}.html")
        plt.figure()
        shap_values = [exp[1] for exp in explanation]
        words = [exp[0] for exp in explanation]
        shap.summary_plot(shap_values, features=words, show=False)
        for i, (word, score) in enumerate(zip(words, shap_values)):
            if score > 0.1:
                plt.annotate(f"High impact: {word} ({score:.2f})",
                             xy=(i, score), xytext=(i, score + 0.05),
                             arrowprops=dict(facecolor='red', shrink=0.05))
        plt.title(f"SHAP Summary for Sample {idx} - {pred_label}")
        plt.savefig(f"shap_bert_annotated_{idx}.png")
        plt.close()
        logger.info(f"BERT SHAP visualization with annotations saved to shap_bert_annotated_{idx}.png")
    except Exception as e:
        logger.error(f"Failed to generate BERT SHAP explanation for sample {idx}: {e}")
        continue

# 1
# =========================
# Token-Level SHAP Summary Plot
# =========================
logger.info("Generating token-level SHAP summary for better interpretability...")

shap_token_values = []
shap_token_words = []

for idx in sample_indices:
    try:
        explanation = explainer_bert(X_test.iloc[idx])
        words = [w for w, _ in explanation]
        shap_vals = [s for _, s in explanation]
        
        shap_token_words.append(words)
        shap_token_values.append(shap_vals)
        
        logger.info(f"Token-level SHAP collected for Sample {idx}")
    except Exception as e:
        logger.warning(f"Token-level SHAP failed for sample {idx}: {e}")
        continue

# Padding to make all sequences equal length
max_len = max(len(x) for x in shap_token_values)
padded_shap = np.array([
    np.pad(x, (0, max_len - len(x)), constant_values=np.nan)
    for x in shap_token_values
])

padded_words = np.array([
    x + [''] * (max_len - len(x))
    for x in shap_token_words
])

# Plot summary with distinct colors for positive/negative SHAP values
plt.figure(figsize=(10, 5))
shap.summary_plot(padded_shap, features=padded_words, show=False, cmap='RdBu_r')
plt.title("SHAP Summary for Important Tokens (BERT)")
plt.tight_layout()
plt.savefig("shap_token_summary.png")
plt.close()

logger.info("Token-level SHAP summary saved to shap_token_summary.png")


# Comparative SHAP Analysis
logger.info("Generating comparative SHAP analysis...")
all_shap_values = []
all_feature_names = []
feature_names_bert = [f"BERT_prob_{c}" for c in le.classes_]
n_features = len(feature_names_bert)

for i, idx in enumerate(sample_indices):
    explanation = explainer_bert(X_test.iloc[idx])
    bert_shap = np.array([score for _, score in explanation])
    bert_padded_shap = np.zeros(n_features)
    n_bert_features = min(len(bert_shap), len(le.classes_))
    bert_padded_shap[:n_bert_features] = bert_shap[:n_bert_features] if n_bert_features > 0 else 0
    all_shap_values.append(bert_padded_shap)
    all_feature_names.extend([f"BERT_{c}" for c in le.classes_])

all_shap_values = np.array(all_shap_values)
all_feature_names = all_feature_names[:n_features * len(sample_indices)]
all_shap_values = all_shap_values / np.max(np.abs(all_shap_values), axis=1, keepdims=True)

plt.figure(figsize=(12, 6))
shap.summary_plot(all_shap_values, feature_names=all_feature_names, show=False)
plt.title("SHAP Summary for BERT")
plt.savefig("shap_bert_summary.png")
plt.close()
logger.info("SHAP summary saved to shap_bert_summary.png")



# 2
# ==================================================
# Token-Level SHAP Summary Plot (Improved)
# ==================================================
logger.info("Generating token-level SHAP summary for better interpretability...")

shap_token_values = []
shap_token_words = []

# Collect SHAP values for sample inputs
for idx in sample_indices:
    try:
        explanation = explainer_bert(X_test.iloc[idx])
        words = [w for w, _ in explanation]
        shap_vals = [s for _, s in explanation]
        
        shap_token_words.append(words)
        shap_token_values.append(shap_vals)
        
        logger.info(f"Token-level SHAP collected for Sample {idx}")
    except Exception as e:
        logger.warning(f"Token-level SHAP failed for sample {idx}: {e}")
        continue

# Flatten for summary plot
flat_shap_values = []
flat_tokens = []

for shap_vals, tokens in zip(shap_token_values, shap_token_words):
    flat_shap_values.extend(shap_vals)
    flat_tokens.extend(tokens)

# Identify top impactful tokens
import pandas as pd
shap_df = pd.DataFrame({'token': flat_tokens, 'shap_value': flat_shap_values})
top_tokens = shap_df.groupby('token')['shap_value'].mean().abs().sort_values(ascending=False).head(20).index.tolist()

# Filter top tokens from each sample
filtered_shap_values = []
filtered_tokens = []

for tokens, shap_vals in zip(shap_token_words, shap_token_values):
    token_shap_pairs = [(t, s) for t, s in zip(tokens, shap_vals) if t in top_tokens]
    if not token_shap_pairs:
        continue
    f_tokens, f_shap_vals = zip(*token_shap_pairs)
    filtered_shap_values.append(np.array(f_shap_vals))
    filtered_tokens.append(list(f_tokens))

# Pad for SHAP plot
max_len = max(len(seq) for seq in filtered_shap_values)
padded_shap = np.array([
    np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan)
    for seq in filtered_shap_values
])
padded_tokens = np.array([
    seq + [''] * (max_len - len(seq))
    for seq in filtered_tokens
])

# Plot and save
plt.figure(figsize=(10, 5))
shap.summary_plot(padded_shap, features=padded_tokens, show=False)
plt.title("SHAP Summary for Important Tokens (BERT)")
plt.tight_layout()
plt.savefig("shap_token_summary.png")
plt.close()
logger.info("Token-level SHAP summary saved to shap_token_summary.png")

# =========================
# Comparative SHAP Analysis (Unchanged, remains meaningful)
# =========================
logger.info("Generating comparative SHAP analysis...")

all_shap_values = []
all_feature_names = []
feature_names_bert = [f"BERT_prob_{c}" for c in le.classes_]
n_features = len(feature_names_bert)

for i, idx in enumerate(sample_indices):
    try:
        explanation = explainer_bert(X_test.iloc[idx])
        bert_shap = np.array([score for _, score in explanation])
        bert_padded_shap = np.zeros(n_features)
        n_bert_features = min(len(bert_shap), len(le.classes_))
        bert_padded_shap[:n_bert_features] = bert_shap[:n_bert_features] if n_bert_features > 0 else 0
        all_shap_values.append(bert_padded_shap)
        all_feature_names.extend(feature_names_bert)
    except Exception as e:
        logger.warning(f"Comparative SHAP failed for sample {idx}: {e}")
        continue

all_shap_values = np.array(all_shap_values)
all_feature_names = all_feature_names[:n_features * len(all_shap_values)]
all_shap_values = all_shap_values / np.max(np.abs(all_shap_values), axis=1, keepdims=True)

plt.figure(figsize=(12, 6))
# shap.summary_plot(all_shap_values, feature_names=all_feature_names, show=False)
shap.summary_plot(all_shap_values, feature_names=all_feature_names, show=False, cmap='RdBu_r')
plt.title("SHAP Summary for BERT Probabilities")
plt.savefig("shap_bert_summary.png")
plt.close()
logger.info("SHAP summary saved to shap_bert_summary.png")