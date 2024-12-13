import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load and Preprocess Dataset
# -------------------------------

# Load the balanced dataset
file_path = 'balanced_goodreads_reviews.json'

with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

# Extract sentences and labels
sentences = []
labels = []

for entry in data:
    for sentence in entry['review_sentences']:
        labels.append(sentence[0])  # Label: 0 or 1
        sentences.append(sentence[1])  # Text

# Convert to DataFrame
df = pd.DataFrame({'text': sentences, 'label': labels})
print(f"Dataset loaded with {len(df)} samples.")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -------------------------------
# Create Custom Dataset Class
# -------------------------------

class SpoilerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)

        return item

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create datasets
train_dataset = SpoilerDataset(X_train, y_train, tokenizer)
test_dataset = SpoilerDataset(X_test, y_test, tokenizer)

# -------------------------------
# Define Evaluation Metrics
# -------------------------------

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# -------------------------------
# Fine-Tune DistilBERT Model
# -------------------------------

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_distilbert',  # Output directory
    num_train_epochs=3,                # Number of training epochs
    per_device_train_batch_size=16,    # Batch size per device during training
    per_device_eval_batch_size=32,     # Batch size for evaluation
    warmup_steps=500,                  # Number of warmup steps
    weight_decay=0.01,                 # Weight decay
    logging_dir='./logs_distilbert',   # Directory for storing logs
    evaluation_strategy="epoch",       # Evaluate every epoch
    save_strategy="epoch",             # Save model every epoch
    load_best_model_at_end=True,       # Load the best model at the end of training
    metric_for_best_model="f1",        # Best model selection metric
    logging_steps=50,                  # Log every 50 steps
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()
print("Training complete!")

# -------------------------------
# Evaluate the Model
# -------------------------------

# Evaluate on the test set
print("Evaluating the model...")
metrics = trainer.evaluate()
print(f"Test Metrics: {metrics}")

# -------------------------------
# Inference on New Data
# -------------------------------

def predict_spoiler(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    encoding = {key: val.to(device) for key, val in encoding.items()}

    outputs = model(**encoding)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()

    return prediction

# Example inference
new_review = "The book reveals a shocking twist about the protagonist!"
prediction = predict_spoiler(new_review)
if prediction == 1:
    print("Prediction: Contains Spoiler")
else:
    print("Prediction: No Spoiler")