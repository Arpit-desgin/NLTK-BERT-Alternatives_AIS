"""
Fine-Tuning BERT for Custom Tasks
==================================
Complete guide to fine-tuning BERT for text classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

print("=" * 80)
print("FINE-TUNING BERT FOR TEXT CLASSIFICATION")
print("=" * 80)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

# ============================================================================
# 1. PREPARE DATASET
# ============================================================================
print("1. PREPARING DATASET")
print("-" * 80)

# Sample dataset: Product reviews (positive/negative)
sample_data = [
    ("This product is amazing! Highly recommend it.", 1),
    ("Terrible quality, broke after one day.", 0),
    ("Best purchase I've ever made!", 1),
    ("Complete waste of money.", 0),
    ("Excellent product, exceeded expectations.", 1),
    ("Very disappointed with this purchase.", 0),
    ("Outstanding quality and fast shipping.", 1),
    ("Poor quality, not worth the price.", 0),
    ("Love it! Will buy again.", 1),
    ("Horrible experience, do not buy.", 0),
    ("Great value for money.", 1),
    ("Defective product, returned it.", 0),
    ("Fantastic! Exactly what I needed.", 1),
    ("Cheap materials, fell apart quickly.", 0),
    ("Superb quality, very satisfied.", 1),
    ("Worst purchase ever made.", 0),
    ("Highly recommended, five stars!", 1),
    ("Total disappointment.", 0),
    ("Perfect product, no complaints.", 1),
    ("Avoid this product at all costs.", 0),
]

# Extend dataset for better training
extended_data = sample_data * 10  # 200 samples

texts = [item[0] for item in extended_data]
labels = [item[1] for item in extended_data]

print(f"Total samples: {len(texts)}")
print(f"Positive samples: {sum(labels)}")
print(f"Negative samples: {len(labels) - sum(labels)}")

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# ============================================================================
# 2. CREATE CUSTOM DATASET CLASS
# ============================================================================
print("\n2. CREATING CUSTOM DATASET")
print("-" * 80)

class TextClassificationDataset(Dataset):
    """Custom Dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

print(f"Dataset created successfully")
print(f"Sample encoding shape: {train_dataset[0]['input_ids'].shape}")

# ============================================================================
# 3. CREATE DATA LOADERS
# ============================================================================
print("\n3. CREATING DATA LOADERS")
print("-" * 80)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Batch size: {BATCH_SIZE}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ============================================================================
# 4. INITIALIZE MODEL
# ============================================================================
print("\n4. INITIALIZING BERT MODEL")
print("-" * 80)

NUM_CLASSES = 2  # Binary classification (positive/negative)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=NUM_CLASSES
)
model.to(device)

print(f"Model: BERT-base-uncased")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# 5. SETUP OPTIMIZER AND SCHEDULER
# ============================================================================
print("\n5. SETTING UP OPTIMIZER AND SCHEDULER")
print("-" * 80)

EPOCHS = 3
LEARNING_RATE = 2e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print(f"Optimizer: AdamW")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Total training steps: {total_steps}")

# ============================================================================
# 6. TRAINING FUNCTION
# ============================================================================
print("\n6. DEFINING TRAINING FUNCTION")
print("-" * 80)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        actual_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    
    return avg_loss, accuracy

# ============================================================================
# 7. EVALUATION FUNCTION
# ============================================================================

def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            actual_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    
    return avg_loss, accuracy, predictions, actual_labels

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
print("\n8. STARTING TRAINING")
print("-" * 80)

best_val_accuracy = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Evaluate
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"✓ New best model saved! (Accuracy: {val_acc:.4f})")

# ============================================================================
# 9. FINAL EVALUATION
# ============================================================================
print("\n\n9. FINAL EVALUATION")
print("-" * 80)

# Load best model
model.load_state_dict(torch.load('best_model.pt'))

val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)

print(f"\nBest Validation Accuracy: {val_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=['Negative', 'Positive']))

# ============================================================================
# 10. INFERENCE FUNCTION
# ============================================================================
print("\n10. INFERENCE ON NEW DATA")
print("-" * 80)

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for new text"""
    model.eval()
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

# Test on new examples
test_examples = [
    "This is the best product I've ever used!",
    "Absolutely terrible, complete waste of money.",
    "It's okay, nothing special but does the job.",
    "Amazing quality and great customer service!",
]

print("Predictions on new data:\n")
for text in test_examples:
    sentiment, confidence = predict_sentiment(text, model, tokenizer, device)
    print(f"Text: {text}")
    print(f"Prediction: {sentiment} (Confidence: {confidence:.4f})\n")

# ============================================================================
# 11. SAVE MODEL FOR PRODUCTION
# ============================================================================
print("\n11. SAVING MODEL FOR PRODUCTION")
print("-" * 80)

# Save model and tokenizer
model.save_pretrained('./fine_tuned_bert_model')
tokenizer.save_pretrained('./fine_tuned_bert_model')

print("✓ Model saved to './fine_tuned_bert_model'")
print("\nTo load the model later:")
print("""
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert_model')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert_model')
""")

# ============================================================================
# 12. HYPERPARAMETER TUNING TIPS
# ============================================================================
print("\n\n12. HYPERPARAMETER TUNING TIPS")
print("-" * 80)

print("""
Key Hyperparameters to Tune:

1. Learning Rate:
   - BERT: 2e-5, 3e-5, 5e-5
   - Smaller datasets: Lower learning rates
   - Larger datasets: Can use higher rates

2. Batch Size:
   - 16 or 32 for most tasks
   - Larger batches: More stable but need more memory
   - Smaller batches: Less memory but more noise

3. Epochs:
   - 2-4 epochs typically sufficient
   - More epochs risk overfitting
   - Use early stopping

4. Max Sequence Length:
   - 128 for short texts (tweets, reviews)
   - 256-512 for longer documents
   - Trade-off: accuracy vs speed

5. Warmup Steps:
   - 0-10% of total training steps
   - Helps stabilize training

6. Weight Decay:
   - 0.01 is common
   - Helps prevent overfitting

7. Dropout:
   - BERT default: 0.1
   - Increase for overfitting (0.2-0.3)
""")

print("\n" + "=" * 80)
print("FINE-TUNING GUIDE COMPLETED")
print("=" * 80)

print("\n📚 SUMMARY:")
print("  ✓ Created custom dataset and data loaders")
print("  ✓ Initialized BERT for sequence classification")
print("  ✓ Set up optimizer (AdamW) and scheduler")
print("  ✓ Trained model for multiple epochs")
print("  ✓ Evaluated performance on validation set")
print("  ✓ Saved best model for production use")
print("  ✓ Created inference function for new predictions")
print("\n🎯 Next Steps:")
print("  • Collect more training data for better performance")
print("  • Experiment with different BERT variants")
print("  • Try data augmentation techniques")
print("  • Deploy model using FastAPI or Flask")
