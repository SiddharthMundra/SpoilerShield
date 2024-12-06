import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, roc_auc_score, roc_curve


# -------------------------------
# Data Preparation
# -------------------------------

# Load the dataset from JSON
# Assumption: Each line in the JSON file is a separate JSON object
# Adjust 'lines' parameter based on actual dataset format
try:
    data = pd.read_json('balanced_goodreads_reviews.json', lines=True)
except ValueError:
    # If not JSON lines, try without 'lines=True'
    data = pd.read_json('balanced_goodreads_reviews.json')

# Display initial data structure
print("Initial Data Sample:")
print(data.head())

# Extract sentences and labels
sentences = []
labels = []

for index, row in data.iterrows():
    for sentence in row['review_sentences']:
        label, text = sentence
        sentences.append(text)
        labels.append(label)

# Create a new DataFrame with sentence-level data
sentence_data = pd.DataFrame({
    'sentence': sentences,
    'is_spoiler': labels
})

# Display the first few entries
print("\nSentence-Level Data Sample:")
print(sentence_data.head())

# Check class distribution
print("\nClass Distribution:")
print(sentence_data['is_spoiler'].value_counts())

# -------------------------------
# Model 1: Very Basic Keyword-Based Approach
# -------------------------------

# Define Spoiler Keywords/Phrases
# Define an Expanded List of Spoiler Keywords/Phrases
spoiler_keywords = [
    # Existing Keywords
    "plot twist",
    "ending was",
    "killer's identity",
    "story ends",
    "revealed the",
    "unexpectedly",
    "character dies",
    "final chapter",
    "murderer revealed",
    "secret is exposed",
    "revealed",
    "killed",
    "main character dies",
    "secondary character dies",
    "finale details",
    "truth comes out",
    "real identity revealed",
    "battle scene",
    "climactic fight",
    "major betrayal",
    "significant loss",
    "key event revealed",
    "critical moment",
    "backstory revealed",
    "origin story revealed",
    "past revealed",
    "history uncovered",
    "relationship ends",
    "affair revealed",
    "partnership dissolved",
    "friendship broken",
    "alliance betrayed",
    "mystery solved",
    "who did it revealed",
    "culprit identified",
    "emotional climax",
    "heart-wrenching moment",
    "tearjerker ending",
    "emotional revelation",
    "surprise ending",
    "unexpected outcome",
    "astonishing revelation",
    "mind-blowing twist",
]

# Function to Detect Spoilers
def detect_spoilers_keyword(text, keywords):
    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            return 1  # Spoiler detected
    return 0  # No spoiler

# Apply Function to Data and assign to 'predicted_spoiler_keyword'
sentence_data['predicted_spoiler_keyword'] = sentence_data['sentence'].apply(
    lambda x: detect_spoilers_keyword(x, spoiler_keywords)
)

# Display the first few entries with predictions
print("\nSample Predictions:")
print(sentence_data[['sentence', 'is_spoiler', 'predicted_spoiler_keyword']].head())

# -------------------------------
# Evaluation Metrics: Precision, False Positive Rate, and False Negative Rate
# -from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, roc_auc_score, roc_curve

# -------------------------------
# Evaluation Metrics: Precision, False Positive Rate, False Negative Rate, and ROC AUC
# -------------------------------

# Compute confusion matrix
cm = confusion_matrix(sentence_data['is_spoiler'], sentence_data['predicted_spoiler_keyword'])

# Extract TN, FP, FN, TP
# Confusion matrix layout:
# [ [TN, FP],
#   [FN, TP] ]
tn, fp, fn, tp = cm.ravel()

# Calculate Precision
precision = precision_score(sentence_data['is_spoiler'], sentence_data['predicted_spoiler_keyword'])

# Calculate False Positive Rate and False Negative Rate
fp_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
fn_rate = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0

# Calculate ROC AUC
roc_auc = roc_auc_score(sentence_data['is_spoiler'], sentence_data['predicted_spoiler_keyword'])

# Print the metrics in an understandable format
print("\nKeyword-Based Approach Performance:")
print(f"- Precision: {precision:.2f}% of flagged reviews actually contain spoilers.")
print(f"- False Positive Rate: {fp_rate:.2f}% of non-spoiler sentences were incorrectly identified as spoilers.")
print(f"- False Negative Rate: {fn_rate:.2f}% of actual spoiler sentences were missed by the model.")
print(f"- ROC AUC: {roc_auc:.2f}")

# -------------------------------
# Visualization: ROC Curve
# -------------------------------

# Generate the ROC Curve
fpr, tpr, thresholds = roc_curve(sentence_data['is_spoiler'], sentence_data['predicted_spoiler_keyword'])

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label=f'Keyword-Based (AUC = {roc_auc:.2f})')
plt.title('ROC Curve for Keyword-Based Spoiler Detection')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# Visualization: Confusion Matrix
# -------------------------------

# Define labels for the confusion matrix
labels = ['No Spoiler', 'Spoiler']

# Plot confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
plt.title('Confusion Matrix for Keyword-Based Spoiler Detection')
plt.show()
