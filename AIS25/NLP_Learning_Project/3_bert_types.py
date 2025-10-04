"""
BERT Types and Variants - Comprehensive Overview
=================================================
Exploring different BERT architectures and their specific use cases.
"""

from transformers import (
    AutoTokenizer, AutoModel,
    BertModel, RobertaModel, DistilBertModel, AlbertModel,
    pipeline
)
import torch

print("=" * 80)
print("BERT TYPES AND VARIANTS")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

# ============================================================================
# 1. BERT-BASE vs BERT-LARGE
# ============================================================================
print("1. BERT-BASE vs BERT-LARGE")
print("-" * 80)

bert_variants = {
    'BERT-Base': {
        'model': 'bert-base-uncased',
        'layers': 12,
        'hidden_size': 768,
        'attention_heads': 12,
        'parameters': '110M',
        'use_case': 'General purpose, faster inference'
    },
    'BERT-Large': {
        'model': 'bert-large-uncased',
        'layers': 24,
        'hidden_size': 1024,
        'attention_heads': 16,
        'parameters': '340M',
        'use_case': 'Higher accuracy, resource-intensive'
    }
}

for name, specs in bert_variants.items():
    print(f"\n{name}:")
    print(f"  Model: {specs['model']}")
    print(f"  Layers: {specs['layers']}")
    print(f"  Hidden Size: {specs['hidden_size']}")
    print(f"  Attention Heads: {specs['attention_heads']}")
    print(f"  Parameters: {specs['parameters']}")
    print(f"  Use Case: {specs['use_case']}")

# ============================================================================
# 2. CASED vs UNCASED
# ============================================================================
print("\n\n2. CASED vs UNCASED MODELS")
print("-" * 80)

# Uncased example
uncased_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
uncased_tokens = uncased_tokenizer.tokenize("Apple Inc. and apple fruit")

# Cased example
cased_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
cased_tokens = cased_tokenizer.tokenize("Apple Inc. and apple fruit")

print(f"Text: 'Apple Inc. and apple fruit'\n")
print(f"Uncased tokens: {uncased_tokens}")
print(f"Cased tokens:   {cased_tokens}")
print(f"\nUse Uncased: General text, case doesn't matter")
print(f"Use Cased:   NER, proper nouns, case-sensitive tasks")

# ============================================================================
# 3. DOMAIN-SPECIFIC BERT MODELS
# ============================================================================
print("\n\n3. DOMAIN-SPECIFIC BERT MODELS")
print("-" * 80)

domain_models = {
    'BioBERT': {
        'model': 'dmis-lab/biobert-v1.1',
        'domain': 'Biomedical',
        'trained_on': 'PubMed abstracts, PMC articles',
        'use_cases': ['Medical NER', 'Biomedical QA', 'Drug discovery']
    },
    'SciBERT': {
        'model': 'allenai/scibert_scivocab_uncased',
        'domain': 'Scientific',
        'trained_on': 'Scientific papers (1.14M)',
        'use_cases': ['Scientific text classification', 'Citation analysis']
    },
    'FinBERT': {
        'model': 'ProsusAI/finbert',
        'domain': 'Financial',
        'trained_on': 'Financial news, reports',
        'use_cases': ['Financial sentiment', 'Risk assessment']
    },
    'LegalBERT': {
        'model': 'nlpaueb/legal-bert-base-uncased',
        'domain': 'Legal',
        'trained_on': 'Legal documents, case law',
        'use_cases': ['Legal document analysis', 'Contract review']
    },
    'ClinicalBERT': {
        'model': 'emilyalsentzer/Bio_ClinicalBERT',
        'domain': 'Clinical',
        'trained_on': 'Clinical notes, medical records',
        'use_cases': ['Clinical NER', 'Patient record analysis']
    }
}

for name, info in domain_models.items():
    print(f"\n{name}:")
    print(f"  Model: {info['model']}")
    print(f"  Domain: {info['domain']}")
    print(f"  Trained on: {info['trained_on']}")
    print(f"  Use cases: {', '.join(info['use_cases'])}")

# ============================================================================
# 4. MULTILINGUAL BERT
# ============================================================================
print("\n\n4. MULTILINGUAL BERT (mBERT)")
print("-" * 80)

multilingual_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

texts = {
    'English': 'Natural language processing',
    'Spanish': 'Procesamiento del lenguaje natural',
    'French': 'Traitement du langage naturel',
    'German': 'Natürliche Sprachverarbeitung',
    'Chinese': '自然语言处理'
}

print("Multilingual BERT supports 104 languages\n")
print("Tokenization examples:")
for lang, text in texts.items():
    tokens = multilingual_tokenizer.tokenize(text)
    print(f"  {lang:10} '{text}' -> {tokens[:5]}...")

print(f"\nUse Cases:")
print(f"  - Cross-lingual transfer learning")
print(f"  - Multilingual sentiment analysis")
print(f"  - Machine translation support")
print(f"  - Zero-shot language understanding")

# ============================================================================
# 5. TASK-SPECIFIC BERT MODELS
# ============================================================================
print("\n\n5. TASK-SPECIFIC BERT MODELS")
print("-" * 80)

task_models = {
    'Sentiment Analysis': {
        'model': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'task': 'Sentiment classification (1-5 stars)',
        'example': 'Product reviews, customer feedback'
    },
    'Question Answering': {
        'model': 'deepset/bert-base-cased-squad2',
        'task': 'Extractive QA',
        'example': 'Customer support, document search'
    },
    'Named Entity Recognition': {
        'model': 'dslim/bert-base-NER',
        'task': 'Entity extraction',
        'example': 'Information extraction, data mining'
    },
    'Text Classification': {
        'model': 'fabriceyhc/bert-base-uncased-ag_news',
        'task': 'News categorization',
        'example': 'Content moderation, topic detection'
    },
    'Paraphrase Detection': {
        'model': 'bert-base-uncased (fine-tuned on MRPC)',
        'task': 'Semantic similarity',
        'example': 'Duplicate detection, plagiarism check'
    }
}

for task, info in task_models.items():
    print(f"\n{task}:")
    print(f"  Model: {info['model']}")
    print(f"  Task: {info['task']}")
    print(f"  Example: {info['example']}")

# ============================================================================
# 6. BERT FOR SEQUENCE CLASSIFICATION (Practical Example)
# ============================================================================
print("\n\n6. BERT FOR SEQUENCE CLASSIFICATION (Example)")
print("-" * 80)

# Using a pre-trained sentiment model
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

reviews = [
    "This product exceeded all my expectations! Absolutely love it!",
    "Terrible quality, broke after one day. Very disappointed.",
    "It's okay, does the job but nothing special.",
    "Good value for money, would recommend to friends."
]

print("Sentiment Classification (1-5 stars):\n")
for review in reviews:
    result = classifier(review)[0]
    stars = result['label'].split()[0]
    confidence = result['score']
    print(f"Review: {review[:50]}...")
    print(f"Rating: {stars} stars (confidence: {confidence:.3f})\n")

# ============================================================================
# 7. BERT FOR TOKEN CLASSIFICATION (NER Example)
# ============================================================================
print("\n7. BERT FOR TOKEN CLASSIFICATION (NER)")
print("-" * 80)

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)

text = "Elon Musk founded SpaceX in California and later acquired Twitter for $44 billion."

entities = ner(text)

print(f"Text: {text}\n")
print("Extracted Entities:")
for entity in entities:
    print(f"  {entity['word']:20} | {entity['entity_group']:10} | Score: {entity['score']:.3f}")

# ============================================================================
# 8. BERT FOR QUESTION ANSWERING (Example)
# ============================================================================
print("\n\n8. BERT FOR QUESTION ANSWERING")
print("-" * 80)

qa = pipeline(
    "question-answering",
    model="deepset/bert-base-cased-squad2",
    device=0 if torch.cuda.is_available() else -1
)

context = """
The Transformer architecture was introduced in the paper 'Attention is All You Need' in 2017.
BERT, which stands for Bidirectional Encoder Representations from Transformers, was developed by
Google AI Language in 2018. It uses the encoder part of the Transformer architecture and is
pre-trained on a large corpus using masked language modeling and next sentence prediction tasks.
"""

questions = [
    "When was the Transformer architecture introduced?",
    "What does BERT stand for?",
    "Who developed BERT?",
    "What tasks is BERT pre-trained on?"
]

print(f"Context: {context[:100]}...\n")

for q in questions:
    answer = qa(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {answer['answer']} (score: {answer['score']:.3f})\n")

# ============================================================================
# 9. BERT MODEL SIZE COMPARISON
# ============================================================================
print("\n9. BERT MODEL SIZE COMPARISON")
print("-" * 80)

model_sizes = [
    {'name': 'BERT-Tiny', 'layers': 2, 'hidden': 128, 'params': '4.4M', 'speed': '⚡⚡⚡⚡⚡'},
    {'name': 'BERT-Mini', 'layers': 4, 'hidden': 256, 'params': '11M', 'speed': '⚡⚡⚡⚡'},
    {'name': 'BERT-Small', 'layers': 4, 'hidden': 512, 'params': '29M', 'speed': '⚡⚡⚡'},
    {'name': 'BERT-Medium', 'layers': 8, 'hidden': 512, 'params': '41M', 'speed': '⚡⚡⚡'},
    {'name': 'BERT-Base', 'layers': 12, 'hidden': 768, 'params': '110M', 'speed': '⚡⚡'},
    {'name': 'BERT-Large', 'layers': 24, 'hidden': 1024, 'params': '340M', 'speed': '⚡'},
]

print(f"{'Model':<15} {'Layers':<8} {'Hidden':<8} {'Params':<10} {'Speed':<10}")
print("-" * 60)
for model in model_sizes:
    print(f"{model['name']:<15} {model['layers']:<8} {model['hidden']:<8} "
          f"{model['params']:<10} {model['speed']:<10}")

print("\nChoosing the right size:")
print("  - Tiny/Mini/Small: Mobile apps, edge devices, real-time inference")
print("  - Medium: Balanced performance for most applications")
print("  - Base: Standard choice for production systems")
print("  - Large: Maximum accuracy, research, offline processing")

# ============================================================================
# 10. BERT USE CASE RECOMMENDATIONS
# ============================================================================
print("\n\n10. BERT USE CASE RECOMMENDATIONS")
print("-" * 80)

use_cases = {
    'Text Classification': {
        'model': 'BERT-Base',
        'examples': ['Spam detection', 'Topic categorization', 'Intent classification'],
        'why': 'Excellent at understanding context for categorization'
    },
    'Named Entity Recognition': {
        'model': 'BERT-Base-Cased or Domain-specific',
        'examples': ['Person/Org extraction', 'Medical entities', 'Legal entities'],
        'why': 'Preserves case information, crucial for proper nouns'
    },
    'Question Answering': {
        'model': 'BERT-Large or SQuAD-tuned',
        'examples': ['Customer support', 'Search engines', 'Virtual assistants'],
        'why': 'Strong reading comprehension capabilities'
    },
    'Sentiment Analysis': {
        'model': 'Fine-tuned BERT or DistilBERT',
        'examples': ['Product reviews', 'Social media monitoring', 'Brand perception'],
        'why': 'Captures nuanced sentiment and context'
    },
    'Semantic Search': {
        'model': 'Sentence-BERT or BERT embeddings',
        'examples': ['Document retrieval', 'Similar question finding', 'Recommendation'],
        'why': 'Creates meaningful semantic representations'
    },
    'Text Summarization': {
        'model': 'BERT + Decoder (BERTSUM)',
        'examples': ['News summarization', 'Document abstracting', 'Email summaries'],
        'why': 'Encoder provides strong understanding for extraction'
    }
}

for use_case, details in use_cases.items():
    print(f"\n{use_case}:")
    print(f"  Recommended Model: {details['model']}")
    print(f"  Examples: {', '.join(details['examples'])}")
    print(f"  Why: {details['why']}")

print("\n" + "=" * 80)
print("BERT TYPES OVERVIEW COMPLETED")
print("=" * 80)
print("\nKey Selection Criteria:")
print("  1. Task type (classification, NER, QA, etc.)")
print("  2. Domain (general, medical, legal, financial)")
print("  3. Language requirements (monolingual vs multilingual)")
print("  4. Resource constraints (model size, inference speed)")
print("  5. Accuracy requirements (base vs large)")
