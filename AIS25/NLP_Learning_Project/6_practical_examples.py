"""
Practical NLP Examples - Real-World Applications
================================================
End-to-end examples for common NLP tasks.
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re

print("=" * 80)
print("PRACTICAL NLP APPLICATIONS")
print("=" * 80)

device = 0 if torch.cuda.is_available() else -1

# ============================================================================
# 1. EMAIL SPAM DETECTION
# ============================================================================
print("\n1. EMAIL SPAM DETECTION")
print("-" * 80)

spam_classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    device=device
)

emails = [
    "Meeting scheduled for tomorrow at 10 AM in conference room",
    "CONGRATULATIONS! You've won $1,000,000! Click here to claim now!!!",
    "Please review the attached quarterly report by Friday",
    "FREE VIAGRA! Limited time offer! Buy now and save 90%!!!"
]

print("Email Spam Detection Results:\n")
for email in emails:
    result = spam_classifier(email[:100])[0]
    label = "🚫 SPAM" if result['label'] == 'spam' else "✅ HAM"
    print(f"{label} ({result['score']:.3f}): {email[:60]}...")

# ============================================================================
# 2. CUSTOMER REVIEW SENTIMENT ANALYSIS
# ============================================================================
print("\n\n2. CUSTOMER REVIEW SENTIMENT ANALYSIS")
print("-" * 80)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

reviews = [
    "This product exceeded my expectations! The quality is outstanding.",
    "Terrible experience. Product broke after 2 days. Very disappointed.",
    "It's okay. Does what it's supposed to do, nothing special.",
    "Amazing! Best purchase I've made this year. Highly recommend!",
    "Poor quality materials. Not worth the price at all."
]

print("Customer Review Sentiment (1-5 stars):\n")
for review in reviews:
    result = sentiment_analyzer(review)[0]
    stars = int(result['label'].split()[0])
    star_display = "⭐" * stars
    print(f"{star_display} ({result['score']:.3f}): {review[:50]}...")

# ============================================================================
# 3. NAMED ENTITY EXTRACTION (Resume Parsing)
# ============================================================================
print("\n\n3. NAMED ENTITY EXTRACTION - Resume Parsing")
print("-" * 80)

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=device
)

resume_text = """
John Smith is a Software Engineer with 5 years of experience. 
He worked at Google in Mountain View, California from 2018 to 2021.
Currently employed at Microsoft in Seattle, Washington.
Graduated from Stanford University with a degree in Computer Science.
Contact: john.smith@email.com, Phone: +1-555-0123
"""

entities = ner_pipeline(resume_text)

print(f"Resume Text:\n{resume_text}\n")
print("Extracted Information:")

# Group entities by type
entity_groups = {}
for entity in entities:
    entity_type = entity['entity_group']
    if entity_type not in entity_groups:
        entity_groups[entity_type] = []
    entity_groups[entity_type].append(entity['word'])

for entity_type, values in entity_groups.items():
    print(f"\n{entity_type}:")
    for value in set(values):
        print(f"  • {value}")

# ============================================================================
# 4. QUESTION ANSWERING SYSTEM
# ============================================================================
print("\n\n4. QUESTION ANSWERING SYSTEM")
print("-" * 80)

qa_system = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=device
)

context = """
Python is a high-level, interpreted programming language created by Guido van Rossum 
and first released in 1991. It emphasizes code readability with its use of significant 
indentation. Python supports multiple programming paradigms including procedural, 
object-oriented, and functional programming. It is widely used in web development, 
data science, artificial intelligence, and automation. The Python Software Foundation 
manages the development of Python.
"""

questions = [
    "Who created Python?",
    "When was Python first released?",
    "What programming paradigms does Python support?",
    "What is Python used for?",
    "Who manages Python development?"
]

print(f"Context: {context[:100]}...\n")
print("Q&A Results:\n")

for question in questions:
    answer = qa_system(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {answer['answer']} (confidence: {answer['score']:.3f})\n")

# ============================================================================
# 5. DOCUMENT SIMILARITY & DUPLICATE DETECTION
# ============================================================================
print("\n5. DOCUMENT SIMILARITY & DUPLICATE DETECTION")
print("-" * 80)

from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Machine learning is a subset of artificial intelligence",
    "AI includes machine learning as one of its components",
    "The weather forecast predicts rain tomorrow",
    "Artificial intelligence encompasses machine learning techniques",
    "Tomorrow's weather will be rainy according to forecasts"
]

embeddings = sbert_model.encode(documents)
similarity_matrix = cosine_similarity(embeddings)

print("Document Similarity Matrix:\n")
print(f"{'':60} ", end='')
for i in range(len(documents)):
    print(f"D{i+1}   ", end='')
print("\n")

for i, doc in enumerate(documents):
    print(f"D{i+1}: {doc[:55]:55} ", end='')
    for j in range(len(documents)):
        sim = similarity_matrix[i][j]
        print(f"{sim:.2f}  ", end='')
    print()

# Find duplicates (similarity > 0.7)
print("\n\nPotential Duplicates (similarity > 0.7):")
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if similarity_matrix[i][j] > 0.7:
            print(f"\nD{i+1} ↔ D{j+1} (similarity: {similarity_matrix[i][j]:.3f})")
            print(f"  • {documents[i]}")
            print(f"  • {documents[j]}")

# ============================================================================
# 6. TEXT SUMMARIZATION
# ============================================================================
print("\n\n6. TEXT SUMMARIZATION")
print("-" * 80)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

article = """
Artificial intelligence (AI) is transforming industries worldwide. From healthcare 
to finance, AI technologies are being deployed to solve complex problems and improve 
efficiency. In healthcare, AI algorithms can analyze medical images to detect diseases 
earlier than human doctors. Financial institutions use AI for fraud detection and 
algorithmic trading. The retail sector leverages AI for personalized recommendations 
and inventory management. However, the rapid adoption of AI also raises concerns about 
job displacement, privacy, and ethical considerations. Experts emphasize the need for 
responsible AI development that considers societal impacts. Governments and organizations 
are working on AI governance frameworks to ensure these technologies benefit humanity 
while minimizing potential risks.
"""

summary = summarizer(article, max_length=60, min_length=30, do_sample=False)[0]

print(f"Original Article ({len(article.split())} words):")
print(f"{article}\n")
print(f"Summary ({len(summary['summary_text'].split())} words):")
print(f"{summary['summary_text']}")

# ============================================================================
# 7. INTENT CLASSIFICATION (Chatbot)
# ============================================================================
print("\n\n7. INTENT CLASSIFICATION - Chatbot")
print("-" * 80)

intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

candidate_labels = [
    "greeting",
    "order_status",
    "product_inquiry",
    "complaint",
    "technical_support",
    "account_management"
]

user_queries = [
    "Hello, how are you?",
    "Where is my order? I placed it 3 days ago.",
    "Do you have this product in blue color?",
    "I'm very unhappy with the service quality",
    "My app keeps crashing when I try to login",
    "I want to update my email address"
]

print("User Intent Classification:\n")
for query in user_queries:
    result = intent_classifier(query, candidate_labels)
    top_intent = result['labels'][0]
    confidence = result['scores'][0]
    print(f"Query: {query}")
    print(f"Intent: {top_intent} (confidence: {confidence:.3f})\n")

# ============================================================================
# 8. KEYWORD EXTRACTION
# ============================================================================
print("\n8. KEYWORD EXTRACTION")
print("-" * 80)

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def extract_keywords(text, top_n=5):
    """Extract keywords using TF-IDF"""
    # Tokenize and clean
    words = re.findall(r'\b[a-z]+\b', text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Calculate word frequency
    word_freq = Counter(words)
    
    return word_freq.most_common(top_n)

text = """
Deep learning is a subset of machine learning that uses neural networks with multiple 
layers. These deep neural networks can learn hierarchical representations of data. 
Convolutional neural networks are particularly effective for image recognition tasks, 
while recurrent neural networks excel at sequence modeling. Transformer architectures 
have revolutionized natural language processing. Training deep learning models requires 
large datasets and significant computational resources, often utilizing GPUs or TPUs.
"""

keywords = extract_keywords(text, top_n=8)

print(f"Text: {text}\n")
print("Extracted Keywords:")
for word, freq in keywords:
    print(f"  • {word}: {freq} occurrences")

# ============================================================================
# 9. LANGUAGE DETECTION
# ============================================================================
print("\n\n9. LANGUAGE DETECTION")
print("-" * 80)

lang_detector = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=device
)

multilingual_texts = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "Guten Tag, wie geht es Ihnen?",
    "Ciao, come stai?",
    "こんにちは、お元気ですか？",
    "你好，你好吗？"
]

print("Language Detection Results:\n")
for text in multilingual_texts:
    result = lang_detector(text)[0]
    lang_code = result['label']
    confidence = result['score']
    
    lang_names = {
        'en': 'English', 'fr': 'French', 'es': 'Spanish',
        'de': 'German', 'it': 'Italian', 'ja': 'Japanese',
        'zh': 'Chinese'
    }
    
    lang_name = lang_names.get(lang_code, lang_code)
    print(f"{lang_name:10} ({confidence:.3f}): {text}")

# ============================================================================
# 10. CONTENT MODERATION
# ============================================================================
print("\n\n10. CONTENT MODERATION - Toxic Comment Detection")
print("-" * 80)

toxicity_detector = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=device
)

comments = [
    "Great post! Thanks for sharing this information.",
    "You're an idiot and don't know what you're talking about!",
    "I respectfully disagree with your opinion on this matter.",
    "This is garbage! Complete waste of time!",
    "Interesting perspective. I'd like to learn more about this."
]

print("Content Moderation Results:\n")
for comment in comments:
    result = toxicity_detector(comment)[0]
    is_toxic = result['label'] == 'toxic'
    status = "🚫 TOXIC" if is_toxic else "✅ SAFE"
    print(f"{status} ({result['score']:.3f}): {comment[:50]}...")

# ============================================================================
# 11. SEMANTIC SEARCH
# ============================================================================
print("\n\n11. SEMANTIC SEARCH ENGINE")
print("-" * 80)

knowledge_base = [
    "Python is a high-level programming language used for web development and data science",
    "Machine learning algorithms can learn patterns from data without explicit programming",
    "Neural networks are inspired by biological neurons in the human brain",
    "Natural language processing enables computers to understand human language",
    "Deep learning uses multiple layers to progressively extract features from data",
    "Computer vision allows machines to interpret and understand visual information",
    "Reinforcement learning trains agents through rewards and penalties"
]

# Encode knowledge base
kb_embeddings = sbert_model.encode(knowledge_base)

def semantic_search(query, top_k=3):
    """Search knowledge base using semantic similarity"""
    query_embedding = sbert_model.encode([query])
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': knowledge_base[idx],
            'score': similarities[idx]
        })
    return results

queries = [
    "What is Python used for?",
    "How do neural networks work?",
    "Tell me about AI learning from data"
]

print("Semantic Search Results:\n")
for query in queries:
    print(f"Query: {query}\n")
    results = semantic_search(query, top_k=2)
    for i, result in enumerate(results, 1):
        print(f"  {i}. (score: {result['score']:.3f}) {result['text']}")
    print()

# ============================================================================
# 12. MULTI-LABEL CLASSIFICATION
# ============================================================================
print("\n12. MULTI-LABEL TEXT CLASSIFICATION")
print("-" * 80)

multi_label_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    multi_label=True
)

article_labels = ["technology", "business", "science", "health", "sports", "politics"]

articles = [
    "Apple announces new AI chip for faster machine learning on mobile devices",
    "Scientists discover breakthrough in cancer treatment using gene therapy",
    "Stock market reaches all-time high as tech companies report strong earnings"
]

print("Multi-Label Article Classification:\n")
for article in articles:
    result = multi_label_classifier(article, article_labels)
    
    print(f"Article: {article}\n")
    print("Labels:")
    for label, score in zip(result['labels'][:3], result['scores'][:3]):
        if score > 0.5:
            print(f"  • {label}: {score:.3f}")
    print()

print("\n" + "=" * 80)
print("PRACTICAL EXAMPLES COMPLETED")
print("=" * 80)

print("\n📊 SUMMARY OF APPLICATIONS:")
print("  ✅ Email Spam Detection")
print("  ✅ Customer Review Sentiment Analysis")
print("  ✅ Named Entity Extraction (Resume Parsing)")
print("  ✅ Question Answering System")
print("  ✅ Document Similarity & Duplicate Detection")
print("  ✅ Text Summarization")
print("  ✅ Intent Classification (Chatbot)")
print("  ✅ Keyword Extraction")
print("  ✅ Language Detection")
print("  ✅ Content Moderation")
print("  ✅ Semantic Search Engine")
print("  ✅ Multi-Label Classification")

print("\n🚀 NEXT STEPS:")
print("  • Adapt these examples to your use case")
print("  • Fine-tune models on your domain data")
print("  • Deploy using FastAPI or Flask")
print("  • Monitor performance in production")
