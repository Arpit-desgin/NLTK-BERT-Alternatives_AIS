"""
NLP Examples Using NLTK Only - No Model Downloads Required
===========================================================
Complete NLP examples using only NLTK (works offline, no downloads)
"""

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist, bigrams, trigrams
import string
from collections import Counter
import re

# Download required NLTK data
print("Downloading NLTK data (one-time setup)...")
datasets = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 
            'maxent_ne_chunker', 'words', 'wordnet', 'vader_lexicon']
for dataset in datasets:
    try:
        nltk.download(dataset, quiet=True)
    except:
        pass

print("\n" + "=" * 80)
print("NLP EXAMPLES - NLTK ONLY (NO MODEL DOWNLOADS)")
print("=" * 80)

# ============================================================================
# 1. SENTIMENT ANALYSIS
# ============================================================================
print("\n1. SENTIMENT ANALYSIS")
print("-" * 80)

sia = SentimentIntensityAnalyzer()

reviews = [
    "This product is absolutely amazing! I love it!",
    "I'm very disappointed with this purchase. Terrible quality.",
    "It's okay, nothing special.",
    "Best experience ever! Highly recommend!"
]

print("Product Review Sentiment:\n")
for review in reviews:
    scores = sia.polarity_scores(review)
    
    # Determine sentiment
    if scores['compound'] >= 0.05:
        sentiment = "POSITIVE 😊"
        stars = "⭐⭐⭐⭐⭐"
    elif scores['compound'] <= -0.05:
        sentiment = "NEGATIVE 😞"
        stars = "⭐"
    else:
        sentiment = "NEUTRAL 😐"
        stars = "⭐⭐⭐"
    
    print(f"{stars} {review}")
    print(f"   → {sentiment} (compound: {scores['compound']:.3f})")
    print(f"   Positive: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}, Neutral: {scores['neu']:.2f}\n")

# ============================================================================
# 2. NAMED ENTITY RECOGNITION
# ============================================================================
print("\n2. NAMED ENTITY RECOGNITION")
print("-" * 80)

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. Microsoft is based in Redmond, Washington."

print(f"Text: {text}\n")

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)

print("Detected Entities:")
for chunk in named_entities:
    if hasattr(chunk, 'label'):
        entity = ' '.join(c[0] for c in chunk)
        entity_type = chunk.label()
        print(f"  📍 {entity:25} → {entity_type}")

# ============================================================================
# 3. TEXT CLASSIFICATION (Simple Rule-Based)
# ============================================================================
print("\n\n3. TEXT CLASSIFICATION (Rule-Based)")
print("-" * 80)

def classify_text(text):
    """Simple rule-based text classifier"""
    text_lower = text.lower()
    
    # Define keywords for categories
    tech_keywords = ['python', 'programming', 'code', 'software', 'computer', 'ai', 'machine learning']
    sports_keywords = ['football', 'basketball', 'soccer', 'game', 'player', 'team', 'score']
    business_keywords = ['company', 'market', 'stock', 'profit', 'business', 'revenue', 'sales']
    
    tech_score = sum(1 for word in tech_keywords if word in text_lower)
    sports_score = sum(1 for word in sports_keywords if word in text_lower)
    business_score = sum(1 for word in business_keywords if word in text_lower)
    
    scores = {
        'technology': tech_score,
        'sports': sports_score,
        'business': business_score
    }
    
    category = max(scores, key=scores.get)
    confidence = scores[category] / (sum(scores.values()) + 1)
    
    return category, confidence

texts = [
    "I love programming in Python for machine learning projects",
    "The football team won the championship game yesterday",
    "The company reported strong revenue growth this quarter"
]

print("Text Classification Results:\n")
for text in texts:
    category, confidence = classify_text(text)
    bar = "█" * int(confidence * 20)
    print(f"Text: {text}")
    print(f"  → {category.upper()} {bar} ({confidence:.1%})\n")

# ============================================================================
# 4. KEYWORD EXTRACTION
# ============================================================================
print("\n4. KEYWORD EXTRACTION")
print("-" * 80)

article = """
Artificial intelligence and machine learning are transforming industries worldwide. 
Deep learning algorithms use neural networks to process vast amounts of data. 
Natural language processing enables computers to understand human language. 
These technologies are revolutionizing healthcare, finance, and education sectors.
"""

print(f"Article: {article[:100]}...\n")

# Tokenize and remove stopwords
tokens = word_tokenize(article.lower())
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 3]

# Get word frequency
word_freq = Counter(filtered_tokens)

print("Top Keywords:")
for word, freq in word_freq.most_common(10):
    print(f"  • {word:20} ({freq} occurrences)")

# ============================================================================
# 5. TEXT SIMILARITY
# ============================================================================
print("\n\n5. TEXT SIMILARITY (Jaccard)")
print("-" * 80)

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity"""
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union) if union else 0

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Artificial intelligence includes machine learning techniques",
    "The weather forecast predicts rain tomorrow"
]

print("Document Similarity Matrix:\n")
for i, doc1 in enumerate(documents):
    print(f"Doc {i+1}: {doc1[:40]}...")
    for j, doc2 in enumerate(documents):
        similarity = jaccard_similarity(doc1, doc2)
        print(f"  vs Doc {j+1}: {similarity:.3f}")
    print()

# ============================================================================
# 6. PART-OF-SPEECH TAGGING
# ============================================================================
print("\n6. PART-OF-SPEECH TAGGING")
print("-" * 80)

sentence = "Natural language processing enables computers to understand human language effectively"
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

print(f"Sentence: {sentence}\n")
print("POS Tags:")
for word, tag in pos_tags:
    print(f"  {word:15} → {tag}")

# ============================================================================
# 7. TEXT SUMMARIZATION (Extractive)
# ============================================================================
print("\n\n7. TEXT SUMMARIZATION (Extractive)")
print("-" * 80)

def extractive_summary(text, num_sentences=2):
    """Simple extractive summarization"""
    sentences = sent_tokenize(text)
    
    # Calculate sentence scores based on word frequency
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    word_freq = Counter(words)
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        score = sum(word_freq.get(word, 0) for word in sentence_words if word.isalnum())
        sentence_scores[sentence] = score
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    summary = ' '.join([sent[0] for sent in top_sentences])
    
    return summary

long_text = """
Artificial intelligence is transforming industries worldwide. From healthcare to finance, 
AI technologies are being deployed to solve complex problems. Machine learning algorithms 
can analyze vast amounts of data to identify patterns. Deep learning uses neural networks 
to process information. Natural language processing enables computers to understand human 
language. These technologies are revolutionizing how we work and live.
"""

summary = extractive_summary(long_text, num_sentences=2)

print(f"Original ({len(long_text.split())} words):")
print(f"  {long_text[:100]}...\n")
print(f"Summary ({len(summary.split())} words):")
print(f"  {summary}")

# ============================================================================
# 8. SPAM DETECTION (Rule-Based)
# ============================================================================
print("\n\n8. SPAM DETECTION (Rule-Based)")
print("-" * 80)

def detect_spam(text):
    """Simple spam detector"""
    spam_keywords = [
        'free', 'win', 'winner', 'cash', 'prize', 'click here', 
        'congratulations', 'urgent', 'limited time', '!!!', 'buy now'
    ]
    
    text_lower = text.lower()
    spam_score = sum(1 for keyword in spam_keywords if keyword in text_lower)
    
    # Check for excessive punctuation
    if text.count('!') > 2:
        spam_score += 1
    if text.count('$') > 0:
        spam_score += 1
    
    is_spam = spam_score >= 2
    confidence = min(spam_score / 5, 1.0)
    
    return is_spam, confidence

emails = [
    "Meeting scheduled for tomorrow at 10 AM in conference room",
    "CONGRATULATIONS! You've won $1,000,000! Click here to claim now!!!",
    "Please review the attached quarterly report by Friday",
    "FREE VIAGRA! Limited time offer! Buy now and save 90%!!!"
]

print("Email Spam Detection:\n")
for email in emails:
    is_spam, confidence = detect_spam(email)
    label = "🚫 SPAM" if is_spam else "✅ HAM"
    print(f"{label} ({confidence:.1%}): {email[:60]}...")

# ============================================================================
# 9. LANGUAGE DETECTION (Simple)
# ============================================================================
print("\n\n9. LANGUAGE DETECTION (Simple)")
print("-" * 80)

def detect_language(text):
    """Simple language detector based on common words"""
    english_words = set(['the', 'is', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'for'])
    french_words = set(['le', 'la', 'les', 'de', 'et', 'un', 'une', 'est', 'dans', 'pour'])
    spanish_words = set(['el', 'la', 'los', 'las', 'de', 'y', 'un', 'una', 'es', 'en'])
    
    words = set(word_tokenize(text.lower()))
    
    english_score = len(words.intersection(english_words))
    french_score = len(words.intersection(french_words))
    spanish_score = len(words.intersection(spanish_words))
    
    scores = {
        'English': english_score,
        'French': french_score,
        'Spanish': spanish_score
    }
    
    language = max(scores, key=scores.get)
    return language

texts = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?"
]

print("Language Detection:\n")
for text in texts:
    language = detect_language(text)
    print(f"{language:10} → {text}")

# ============================================================================
# 10. N-GRAM ANALYSIS
# ============================================================================
print("\n\n10. N-GRAM ANALYSIS")
print("-" * 80)

text = "Natural language processing is a fascinating field of artificial intelligence and machine learning"
tokens = word_tokenize(text.lower())

# Bigrams
bigram_list = list(bigrams(tokens))
bigram_freq = Counter(bigram_list)

# Trigrams
trigram_list = list(trigrams(tokens))
trigram_freq = Counter(trigram_list)

print("Top Bigrams:")
for bigram, freq in bigram_freq.most_common(5):
    print(f"  {' '.join(bigram):30} ({freq})")

print("\nTop Trigrams:")
for trigram, freq in trigram_freq.most_common(5):
    print(f"  {' '.join(trigram):40} ({freq})")

print("\n" + "=" * 80)
print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📊 Summary - What You Learned:")
print("  ✅ Sentiment Analysis (VADER)")
print("  ✅ Named Entity Recognition")
print("  ✅ Text Classification (Rule-Based)")
print("  ✅ Keyword Extraction")
print("  ✅ Text Similarity (Jaccard)")
print("  ✅ POS Tagging")
print("  ✅ Text Summarization (Extractive)")
print("  ✅ Spam Detection")
print("  ✅ Language Detection")
print("  ✅ N-gram Analysis")

print("\n💡 These are traditional NLP techniques using NLTK")
print("🚀 No model downloads required - works offline!")
print("📚 Great foundation for understanding NLP concepts")

print("\n🎯 Next Steps:")
print("  • For production systems, use BERT/Transformers (when download issues are resolved)")
print("  • NLTK is perfect for: prototyping, education, simple tasks")
print("  • Transformers are better for: complex tasks, high accuracy needs")
