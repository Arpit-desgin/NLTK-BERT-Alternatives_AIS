"""
BERT Alternatives - Comprehensive Comparison
=============================================
Exploring modern alternatives to BERT with implementations and comparisons.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModel,
    RobertaTokenizer, RobertaModel,
    DistilBertTokenizer, DistilBertModel,
    AlbertTokenizer, AlbertModel,
    XLNetTokenizer, XLNetModel,
    ElectraTokenizer, ElectraModel,
    pipeline
)
import time
import numpy as np

print("=" * 80)
print("BERT ALTERNATIVES - COMPREHENSIVE GUIDE")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

# ============================================================================
# 1. RoBERTa (Robustly Optimized BERT)
# ============================================================================
print("1. RoBERTa - Robustly Optimized BERT Approach")
print("-" * 80)

print("""
Key Improvements over BERT:
  ✓ Trained on 10x more data (160GB vs 16GB)
  ✓ Removed Next Sentence Prediction (NSP) task
  ✓ Dynamic masking instead of static
  ✓ Larger batch sizes and longer sequences
  ✓ Better performance on downstream tasks

Use Cases:
  - Text classification with higher accuracy
  - Sentiment analysis
  - Natural language inference
  - Question answering
""")

# RoBERTa Example
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')
roberta_model.to(device)
roberta_model.eval()

text = "RoBERTa improves upon BERT's training methodology"
inputs = roberta_tokenizer(text, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = roberta_model(**inputs)

print(f"\nExample: '{text}'")
print(f"Output shape: {outputs.last_hidden_state.shape}")
print(f"Parameters: ~125M (base), ~355M (large)")

# ============================================================================
# 2. DistilBERT (Distilled BERT)
# ============================================================================
print("\n\n2. DistilBERT - Lightweight and Fast")
print("-" * 80)

print("""
Key Features:
  ✓ 40% smaller than BERT-base
  ✓ 60% faster inference
  ✓ Retains 97% of BERT's performance
  ✓ Uses knowledge distillation
  ✓ 6 layers instead of 12

Perfect For:
  - Mobile applications
  - Real-time inference
  - Resource-constrained environments
  - Edge computing
""")

# DistilBERT Example
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert_model.to(device)
distilbert_model.eval()

text = "DistilBERT is fast and efficient"
inputs = distilbert_tokenizer(text, return_tensors='pt').to(device)

start_time = time.time()
with torch.no_grad():
    outputs = distilbert_model(**inputs)
inference_time = time.time() - start_time

print(f"\nExample: '{text}'")
print(f"Output shape: {outputs.last_hidden_state.shape}")
print(f"Inference time: {inference_time*1000:.2f}ms")
print(f"Parameters: ~66M (vs BERT's 110M)")

# ============================================================================
# 3. ALBERT (A Lite BERT)
# ============================================================================
print("\n\n3. ALBERT - A Lite BERT")
print("-" * 80)

print("""
Key Innovations:
  ✓ Factorized embedding parameterization
  ✓ Cross-layer parameter sharing
  ✓ 18x fewer parameters than BERT-large
  ✓ Sentence Order Prediction (SOP) instead of NSP
  ✓ Better performance with fewer parameters

Advantages:
  - Extremely parameter-efficient
  - Scales well to larger models
  - Lower memory footprint
  - Faster training
""")

# ALBERT Example
albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
albert_model = AlbertModel.from_pretrained('albert-base-v2')
albert_model.to(device)
albert_model.eval()

text = "ALBERT achieves efficiency through parameter sharing"
inputs = albert_tokenizer(text, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = albert_model(**inputs)

print(f"\nExample: '{text}'")
print(f"Output shape: {outputs.last_hidden_state.shape}")
print(f"Parameters: ~12M (base) vs BERT's 110M")

# ============================================================================
# 4. XLNet
# ============================================================================
print("\n\n4. XLNet - Generalized Autoregressive Pretraining")
print("-" * 80)

print("""
Key Differences from BERT:
  ✓ Permutation language modeling (not masked LM)
  ✓ Autoregressive approach
  ✓ Captures bidirectional context without masking
  ✓ Transformer-XL architecture
  ✓ Better on long sequences

Strengths:
  - Superior performance on many benchmarks
  - No pretrain-finetune discrepancy
  - Better at modeling dependencies
  - Excellent for long documents
""")

# XLNet Example (loading only, as it's larger)
print(f"\nXLNet Architecture:")
print(f"  - Uses permutation-based training")
print(f"  - Segment recurrence mechanism")
print(f"  - Relative positional encoding")
print(f"  - Parameters: ~110M (base), ~340M (large)")

# ============================================================================
# 5. ELECTRA (Efficiently Learning an Encoder)
# ============================================================================
print("\n\n5. ELECTRA - Efficiently Learning an Encoder")
print("-" * 80)

print("""
Novel Approach:
  ✓ Replaced Token Detection (RTD) instead of MLM
  ✓ Generator-Discriminator architecture
  ✓ More sample-efficient training
  ✓ Better performance with less compute
  ✓ Learns from all tokens, not just masked ones

Benefits:
  - 30x more compute-efficient than BERT
  - Better performance on small models
  - Faster convergence
  - Excellent for low-resource scenarios
""")

# ELECTRA Example
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator')
electra_model.to(device)
electra_model.eval()

text = "ELECTRA uses replaced token detection"
inputs = electra_tokenizer(text, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = electra_model(**inputs)

print(f"\nExample: '{text}'")
print(f"Output shape: {outputs.last_hidden_state.shape}")
print(f"Available sizes: Small (14M), Base (110M), Large (335M)")

# ============================================================================
# 6. DeBERTa (Decoding-enhanced BERT)
# ============================================================================
print("\n\n6. DeBERTa - Decoding-enhanced BERT with Disentangled Attention")
print("-" * 80)

print("""
Key Innovations:
  ✓ Disentangled attention mechanism
  ✓ Enhanced mask decoder
  ✓ Separate content and position embeddings
  ✓ Virtual adversarial training
  ✓ State-of-the-art on SuperGLUE

Advantages:
  - Better position encoding
  - Improved attention mechanism
  - Superior performance on NLU tasks
  - Efficient training
  
Model: microsoft/deberta-base (86M params)
""")

# ============================================================================
# 7. Sentence-BERT (SBERT)
# ============================================================================
print("\n\n7. Sentence-BERT - Semantic Similarity Specialist")
print("-" * 80)

print("""
Purpose-Built For:
  ✓ Sentence embeddings
  ✓ Semantic similarity
  ✓ Clustering
  ✓ Information retrieval
  ✓ 1000x faster than BERT for similarity

Architecture:
  - Siamese/triplet network structure
  - Mean/max pooling of token embeddings
  - Optimized for cosine similarity
  - Pre-trained on NLI datasets
""")

# SBERT Example
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is a subset of AI",
    "Artificial intelligence includes machine learning",
    "The weather is nice today"
]

embeddings = sbert_model.encode(sentences)

print(f"\nSentence Embeddings:")
print(f"Shape: {embeddings.shape}")

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)

print(f"\nSimilarity Matrix:")
for i, sent1 in enumerate(sentences):
    print(f"\nS{i+1}: {sent1[:40]}...")
    for j, sent2 in enumerate(sentences):
        print(f"  vs S{j+1}: {similarities[i][j]:.3f}")

# ============================================================================
# 8. T5 (Text-to-Text Transfer Transformer)
# ============================================================================
print("\n\n8. T5 - Text-to-Text Transfer Transformer")
print("-" * 80)

print("""
Unified Framework:
  ✓ All NLP tasks as text-to-text
  ✓ Encoder-decoder architecture
  ✓ Trained on C4 dataset (750GB)
  ✓ Versatile for multiple tasks
  ✓ Easy task specification with prefixes

Tasks Supported:
  - Translation: "translate English to German: text"
  - Summarization: "summarize: text"
  - QA: "question: ... context: ..."
  - Classification: "classify sentiment: text"
  
Sizes: T5-small (60M) to T5-11B (11B params)
""")

# ============================================================================
# 9. GPT-Style Models (Decoder-only)
# ============================================================================
print("\n\n9. GPT-Style Models - Decoder-Only Architecture")
print("-" * 80)

print("""
GPT Series (OpenAI):
  ✓ Autoregressive language models
  ✓ Unidirectional (left-to-right)
  ✓ Excellent for text generation
  ✓ Few-shot learning capabilities
  ✓ GPT-2 (1.5B), GPT-3 (175B), GPT-4

Alternatives:
  - GPT-Neo/GPT-J (EleutherAI): Open-source GPT-3 alternatives
  - BLOOM: Multilingual (176B params)
  - LLaMA: Meta's efficient models
  - Falcon: High-performance open models

Best For:
  - Text generation
  - Creative writing
  - Code generation
  - Conversational AI
""")

# ============================================================================
# 10. PERFORMANCE COMPARISON
# ============================================================================
print("\n\n10. PERFORMANCE COMPARISON")
print("-" * 80)

comparison_data = [
    {
        'Model': 'BERT-Base',
        'Params': '110M',
        'Speed': '⚡⚡',
        'Accuracy': '⭐⭐⭐⭐',
        'Memory': '440MB',
        'Best For': 'General NLU tasks'
    },
    {
        'Model': 'RoBERTa',
        'Params': '125M',
        'Speed': '⚡⚡',
        'Accuracy': '⭐⭐⭐⭐⭐',
        'Memory': '500MB',
        'Best For': 'High accuracy NLU'
    },
    {
        'Model': 'DistilBERT',
        'Params': '66M',
        'Speed': '⚡⚡⚡⚡',
        'Accuracy': '⭐⭐⭐⭐',
        'Memory': '260MB',
        'Best For': 'Fast inference'
    },
    {
        'Model': 'ALBERT',
        'Params': '12M',
        'Speed': '⚡⚡⚡',
        'Accuracy': '⭐⭐⭐⭐',
        'Memory': '48MB',
        'Best For': 'Low memory'
    },
    {
        'Model': 'ELECTRA',
        'Params': '110M',
        'Speed': '⚡⚡⚡',
        'Accuracy': '⭐⭐⭐⭐⭐',
        'Memory': '440MB',
        'Best For': 'Efficient training'
    },
    {
        'Model': 'DeBERTa',
        'Params': '86M',
        'Speed': '⚡⚡',
        'Accuracy': '⭐⭐⭐⭐⭐',
        'Memory': '344MB',
        'Best For': 'SOTA performance'
    },
    {
        'Model': 'Sentence-BERT',
        'Params': '22M',
        'Speed': '⚡⚡⚡⚡⚡',
        'Accuracy': '⭐⭐⭐⭐',
        'Memory': '90MB',
        'Best For': 'Similarity tasks'
    },
    {
        'Model': 'T5-Base',
        'Params': '220M',
        'Speed': '⚡⚡',
        'Accuracy': '⭐⭐⭐⭐⭐',
        'Memory': '850MB',
        'Best For': 'Multi-task learning'
    }
]

print(f"{'Model':<15} {'Params':<10} {'Speed':<10} {'Accuracy':<12} {'Memory':<10} {'Best For':<20}")
print("-" * 95)
for model in comparison_data:
    print(f"{model['Model']:<15} {model['Params']:<10} {model['Speed']:<10} "
          f"{model['Accuracy']:<12} {model['Memory']:<10} {model['Best For']:<20}")

# ============================================================================
# 11. PRACTICAL RECOMMENDATIONS
# ============================================================================
print("\n\n11. MODEL SELECTION GUIDE")
print("-" * 80)

recommendations = {
    'Production Systems (Balanced)': {
        'models': ['BERT-Base', 'RoBERTa-Base'],
        'reason': 'Good balance of accuracy and speed'
    },
    'Mobile/Edge Devices': {
        'models': ['DistilBERT', 'ALBERT-Base', 'MobileBERT'],
        'reason': 'Low memory footprint, fast inference'
    },
    'Maximum Accuracy': {
        'models': ['DeBERTa-Large', 'RoBERTa-Large', 'ELECTRA-Large'],
        'reason': 'State-of-the-art performance'
    },
    'Semantic Search/Similarity': {
        'models': ['Sentence-BERT', 'SimCSE'],
        'reason': 'Optimized for similarity tasks'
    },
    'Multi-task Applications': {
        'models': ['T5', 'BART'],
        'reason': 'Unified text-to-text framework'
    },
    'Low-Resource Training': {
        'models': ['ELECTRA', 'ALBERT'],
        'reason': 'Sample-efficient, parameter-efficient'
    },
    'Multilingual Applications': {
        'models': ['mBERT', 'XLM-RoBERTa', 'mT5'],
        'reason': 'Cross-lingual capabilities'
    },
    'Long Documents': {
        'models': ['Longformer', 'BigBird', 'XLNet'],
        'reason': 'Extended context windows'
    }
}

for use_case, info in recommendations.items():
    print(f"\n{use_case}:")
    print(f"  Recommended: {', '.join(info['models'])}")
    print(f"  Reason: {info['reason']}")

# ============================================================================
# 12. PRACTICAL EXAMPLE - Sentiment Analysis Comparison
# ============================================================================
print("\n\n12. PRACTICAL COMPARISON - Sentiment Analysis")
print("-" * 80)

test_text = "This product is absolutely amazing! Best purchase ever!"

models_to_test = [
    ("BERT", "distilbert-base-uncased-finetuned-sst-2-english"),
    ("RoBERTa", "cardiffnlp/twitter-roberta-base-sentiment"),
]

print(f"Test Text: '{test_text}'\n")

for model_name, model_id in models_to_test:
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        
        start = time.time()
        result = classifier(test_text)[0]
        inference_time = (time.time() - start) * 1000
        
        print(f"{model_name}:")
        print(f"  Sentiment: {result['label']}")
        print(f"  Confidence: {result['score']:.4f}")
        print(f"  Inference Time: {inference_time:.2f}ms\n")
    except Exception as e:
        print(f"{model_name}: Error - {str(e)}\n")

print("\n" + "=" * 80)
print("BERT ALTERNATIVES OVERVIEW COMPLETED")
print("=" * 80)

print("\n📊 SUMMARY:")
print("  • RoBERTa: Better training → Higher accuracy")
print("  • DistilBERT: Smaller, faster → Production-ready")
print("  • ALBERT: Parameter sharing → Memory efficient")
print("  • ELECTRA: Better pretraining → Sample efficient")
print("  • DeBERTa: Better attention → SOTA results")
print("  • Sentence-BERT: Optimized embeddings → Similarity tasks")
print("  • T5: Text-to-text → Multi-task versatility")
print("\n🎯 Choose based on: Task, Resources, Accuracy needs, Inference speed")
