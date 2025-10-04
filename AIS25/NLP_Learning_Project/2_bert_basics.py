"""
BERT (Bidirectional Encoder Representations from Transformers) - Comprehensive Examples
========================================================================================
BERT is a transformer-based model designed to pre-train deep bidirectional representations.
"""

import torch
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    BertForQuestionAnswering, BertForMaskedLM,
    pipeline
)
import numpy as np
import os

# Set environment variable to avoid TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("=" * 80)
print("BERT COMPREHENSIVE EXAMPLES")
print("=" * 80)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 1. BERT TOKENIZATION
print("\n1. BERT TOKENIZATION")
print("-" * 80)

try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=False, resume_download=False)
except:
    print("Downloading BERT tokenizer for the first time...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)

sample_text = "Natural Language Processing with BERT is powerful!"
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text, add_special_tokens=True)

print(f"Original text: {sample_text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Decoded: {tokenizer.decode(token_ids)}")

# Special tokens
print(f"\nSpecial Tokens:")
print(f"  [CLS] token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"  [SEP] token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"  [PAD] token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  [MASK] token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

# 2. BERT EMBEDDINGS
print("\n2. BERT EMBEDDINGS (Contextual Word Representations)")
print("-" * 80)

try:
    model = BertModel.from_pretrained('bert-base-uncased', force_download=False, resume_download=False)
except:
    print("Downloading BERT model for the first time...")
    model = BertModel.from_pretrained('bert-base-uncased', force_download=True)
    
model.to(device)
model.eval()

text = "The bank by the river has great views"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    
last_hidden_states = outputs.last_hidden_state
pooler_output = outputs.pooler_output

print(f"Input text: {text}")
print(f"Last hidden state shape: {last_hidden_states.shape}")
print(f"  (batch_size, sequence_length, hidden_size)")
print(f"Pooler output shape: {pooler_output.shape}")
print(f"  (batch_size, hidden_size)")

# Get word embeddings
word_embeddings = last_hidden_states[0]
print(f"\nWord embeddings for each token:")
for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
    embedding = word_embeddings[i]
    print(f"  {token:10} -> Vector of size {embedding.shape[0]}")

# 3. MASKED LANGUAGE MODELING (MLM)
print("\n3. MASKED LANGUAGE MODELING")
print("-" * 80)

mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.to(device)
mlm_model.eval()

def predict_masked_word(text):
    """Predict masked word in text"""
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = mlm_model(**inputs)
    
    # Get predictions for masked token
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    mask_token_logits = outputs.logits[0, mask_token_index, :]
    
    # Get top 5 predictions
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    return [tokenizer.decode([token]) for token in top_5_tokens]

masked_sentences = [
    "The capital of France is [MASK].",
    "I love to [MASK] music.",
    "Python is a [MASK] language."
]

for sentence in masked_sentences:
    predictions = predict_masked_word(sentence)
    print(f"\nSentence: {sentence}")
    print(f"Top 5 predictions: {predictions}")

# 4. SENTENCE SIMILARITY
print("\n4. SENTENCE SIMILARITY (Using BERT Embeddings)")
print("-" * 80)

def get_sentence_embedding(text):
    """Get sentence embedding using BERT"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding as sentence representation
    return outputs.pooler_output[0].cpu().numpy()

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

sentences = [
    "I love machine learning",
    "I enjoy artificial intelligence",
    "The weather is nice today"
]

embeddings = [get_sentence_embedding(sent) for sent in sentences]

print("Sentence Similarity Matrix:")
print(f"\n{'':30} ", end='')
for i, sent in enumerate(sentences):
    print(f"S{i+1:1}    ", end='')
print()

for i, sent1 in enumerate(sentences):
    print(f"S{i+1}: {sent1[:27]:27} ", end='')
    for j, sent2 in enumerate(sentences):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{similarity:.3f}  ", end='')
    print()

# 5. SENTIMENT ANALYSIS (Using Pipeline)
print("\n5. SENTIMENT ANALYSIS (BERT-based)")
print("-" * 80)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

test_texts = [
    "This movie is absolutely fantastic!",
    "I'm very disappointed with this product.",
    "The service was okay, nothing special.",
    "Best experience ever! Highly recommended!"
]

print("Sentiment Analysis Results:")
for text in test_texts:
    result = sentiment_pipeline(text)[0]
    print(f"\n  Text: {text}")
    print(f"  Sentiment: {result['label']}, Confidence: {result['score']:.4f}")

# 6. QUESTION ANSWERING
print("\n6. QUESTION ANSWERING")
print("-" * 80)

qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad",
    device=0 if torch.cuda.is_available() else -1
)

context = """
BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google in 2018.
It revolutionized NLP by using bidirectional training of Transformer models.
BERT achieved state-of-the-art results on eleven natural language processing tasks.
The model comes in two sizes: BERT-Base (12 layers, 110M parameters) and BERT-Large (24 layers, 340M parameters).
"""

questions = [
    "When was BERT introduced?",
    "Who introduced BERT?",
    "How many layers does BERT-Base have?",
    "What does BERT stand for?"
]

print(f"Context: {context[:100]}...\n")

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (confidence: {result['score']:.4f})\n")

# 7. TEXT CLASSIFICATION (Custom Example)
print("\n7. TEXT CLASSIFICATION SETUP")
print("-" * 80)

print("""
For custom text classification with BERT:

1. Load pre-trained BERT model:
   model = BertForSequenceClassification.from_pretrained(
       'bert-base-uncased',
       num_labels=num_classes
   )

2. Prepare your dataset with labels

3. Fine-tune the model:
   - Use AdamW optimizer
   - Learning rate: 2e-5 to 5e-5
   - Batch size: 16 or 32
   - Epochs: 2-4

4. Training loop:
   for epoch in epochs:
       for batch in train_dataloader:
           outputs = model(**batch)
           loss = outputs.loss
           loss.backward()
           optimizer.step()

5. Evaluate on test set
""")

# 8. NAMED ENTITY RECOGNITION (NER)
print("\n8. NAMED ENTITY RECOGNITION")
print("-" * 80)

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)

ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. Microsoft is based in Redmond, Washington."

entities = ner_pipeline(ner_text)

print(f"Text: {ner_text}\n")
print("Detected Entities:")
for entity in entities:
    print(f"  {entity['word']:20} -> {entity['entity_group']:10} (score: {entity['score']:.4f})")

# 9. FEATURE EXTRACTION
print("\n9. FEATURE EXTRACTION")
print("-" * 80)

feature_extractor = pipeline(
    "feature-extraction",
    model="bert-base-uncased",
    device=0 if torch.cuda.is_available() else -1
)

text = "BERT creates contextual embeddings"
features = feature_extractor(text)

print(f"Text: {text}")
print(f"Feature shape: {np.array(features).shape}")
print(f"  (batch_size, sequence_length, hidden_size)")
print(f"\nThese features can be used for:")
print("  - Downstream classification tasks")
print("  - Clustering similar texts")
print("  - Semantic search")
print("  - Text similarity comparison")

# 10. BERT VARIANTS OVERVIEW
print("\n10. BERT MODEL VARIANTS")
print("-" * 80)

variants = {
    'bert-base-uncased': '12 layers, 110M params, uncased',
    'bert-large-uncased': '24 layers, 340M params, uncased',
    'bert-base-cased': '12 layers, 110M params, preserves case',
    'bert-base-multilingual-cased': 'Supports 104 languages',
    'bert-base-chinese': 'Chinese language support',
}

print("Available BERT Variants:")
for variant, description in variants.items():
    print(f"  {variant:30} -> {description}")

print("\n" + "=" * 80)
print("BERT EXAMPLES COMPLETED")
print("=" * 80)
print("\nKey Takeaways:")
print("  1. BERT uses bidirectional context for better understanding")
print("  2. Pre-trained on large corpora, fine-tunable for specific tasks")
print("  3. Excellent for: classification, NER, QA, similarity")
print("  4. Trade-off: High accuracy vs computational cost")
