import time
import random

# Simulate terminal-like output
def simulate_output():
    # Step 1: Preprocessing
    print("Preprocessing: 0%", end="\r")
    for i in range(1, 101):
        time.sleep(0.03)  # Simulate time delay
        print(f"Preprocessing: {i}%  [1139448/1139448] [00:{i//10:02}:00, {random.randint(90000, 120000)}sentence/s]", end="\r")
    print("\nBatch preprocessing completed.")
    print("Text preprocessing completed.")
    
    # Step 2: Splitting Data
    print("Balancing the dataset...")
    time.sleep(1)
    print("Balanced dataset size: 1139448")
    print("Splitting data into training and testing sets...")
    time.sleep(1)
    print("Data splitting completed.")
    
    # Step 3: Tokenization
    print("Tokenizing text for DistilBERT...")
    time.sleep(1)
    print("Tokenization completed.")
    
    # Step 4: Model Training
    print("\n=== Training DistilBERT Model ===")
    print("Initializing DistilBERT tokenizer and model...")
    time.sleep(1)
    print("Loading pre-trained weights for DistilBERT...")
    time.sleep(1)
    print("Fine-tuning DistilBERT on the dataset...")
    for i in range(1, 101, 10):
        time.sleep(0.5)
        print(f"Training Progress: {i}%  [{i * 10000}/{1139448} steps] [00:{i//10:02}:00, {random.randint(50, 100)}it/s]", end="\r")
    print("\nTraining completed.")
    
    # Step 5: Model Evaluation
    print("\n=== Evaluating DistilBERT Model ===")
    time.sleep(1)
    print("Making predictions on the test set...")
    time.sleep(2)
    print("Predictions completed.")
    
    # Simulated Metrics
    precision = 0.8401
    fp_rate = 0.1521
    fn_rate = 0.1512
    roc_auc = 0.9134
    
    print("\nEvaluation Metrics:")
    print(f"Precision (Spoilers): {precision:.4f}")
    print(f"False Positive Rate (FPR): {fp_rate:.4f}")
    print(f"False Negative Rate (FNR): {fn_rate:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nDetailed Evaluation Completed.")
    
simulate_output()