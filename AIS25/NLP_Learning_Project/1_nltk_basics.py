"""
NLTK (Natural Language Toolkit) - Comprehensive Examples
=========================================================
NLTK is a leading platform for building Python programs to work with human language data.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import string

# Download required NLTK data
def download_nltk_data():
    """Download all necessary NLTK datasets"""
    datasets = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 
                'maxent_ne_chunker', 'words', 'wordnet', 'vader_lexicon']
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
        except:
            print(f"Could not download {dataset}")

download_nltk_data()

# Sample text for demonstrations
sample_text = """
Natural Language Processing (NLP) is a fascinating field of artificial intelligence. 
It enables computers to understand, interpret, and generate human language. 
Companies like Google, Microsoft, and Amazon are heavily investing in NLP technologies.
The applications range from chatbots to machine translation and sentiment analysis.
"""

print("=" * 80)
print("NLTK COMPREHENSIVE EXAMPLES")
print("=" * 80)

# 1. TOKENIZATION
print("\n1. TOKENIZATION")
print("-" * 80)

# Sentence tokenization
sentences = sent_tokenize(sample_text)
print(f"Number of sentences: {len(sentences)}")
print(f"First sentence: {sentences[0]}")

# Word tokenization
words = word_tokenize(sample_text)
print(f"\nNumber of words: {len(words)}")
print(f"First 10 words: {words[:10]}")

# 2. STOPWORD REMOVAL
print("\n2. STOPWORD REMOVAL")
print("-" * 80)

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words 
                  and word not in string.punctuation]
print(f"Original word count: {len(words)}")
print(f"After stopword removal: {len(filtered_words)}")
print(f"Filtered words: {filtered_words[:15]}")

# 3. STEMMING
print("\n3. STEMMING (Porter Stemmer)")
print("-" * 80)

stemmer = PorterStemmer()
test_words = ['running', 'runs', 'ran', 'easily', 'fairly', 'computing', 'computed']
stemmed_words = [(word, stemmer.stem(word)) for word in test_words]
print("Word -> Stem:")
for original, stemmed in stemmed_words:
    print(f"  {original:12} -> {stemmed}")

# 4. LEMMATIZATION
print("\n4. LEMMATIZATION")
print("-" * 80)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [(word, lemmatizer.lemmatize(word, pos='v')) for word in test_words]
print("Word -> Lemma:")
for original, lemma in lemmatized_words:
    print(f"  {original:12} -> {lemma}")

# 5. PART-OF-SPEECH (POS) TAGGING
print("\n5. PART-OF-SPEECH TAGGING")
print("-" * 80)

sample_sentence = "Natural Language Processing enables computers to understand human language"
tokens = word_tokenize(sample_sentence)
pos_tags = pos_tag(tokens)
print("Word -> POS Tag:")
for word, tag in pos_tags:
    print(f"  {word:12} -> {tag}")

# 6. NAMED ENTITY RECOGNITION (NER)
print("\n6. NAMED ENTITY RECOGNITION")
print("-" * 80)

ner_sentence = "Google and Microsoft are competing in AI. Sundar Pichai leads Google in California."
tokens = word_tokenize(ner_sentence)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)

print("Named Entities:")
for chunk in named_entities:
    if hasattr(chunk, 'label'):
        entity = ' '.join(c[0] for c in chunk)
        entity_type = chunk.label()
        print(f"  {entity:20} -> {entity_type}")

# 7. SENTIMENT ANALYSIS (VADER)
print("\n7. SENTIMENT ANALYSIS (VADER)")
print("-" * 80)

sia = SentimentIntensityAnalyzer()

test_sentences = [
    "This product is absolutely amazing! I love it!",
    "This is the worst experience I've ever had.",
    "The product is okay, nothing special.",
    "I'm not sure how I feel about this."
]

print("Sentence -> Sentiment Scores:")
for sentence in test_sentences:
    scores = sia.polarity_scores(sentence)
    print(f"\n  '{sentence}'")
    print(f"  Positive: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}, "
          f"Neutral: {scores['neu']:.2f}, Compound: {scores['compound']:.2f}")
    
    # Determine overall sentiment
    if scores['compound'] >= 0.05:
        sentiment = "POSITIVE"
    elif scores['compound'] <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    print(f"  Overall: {sentiment}")

# 8. FREQUENCY DISTRIBUTION
print("\n8. FREQUENCY DISTRIBUTION")
print("-" * 80)

from nltk import FreqDist

fdist = FreqDist(filtered_words)
print("Top 10 most common words:")
for word, frequency in fdist.most_common(10):
    print(f"  {word:15} : {frequency}")

# 9. BIGRAMS AND TRIGRAMS
print("\n9. BIGRAMS AND TRIGRAMS")
print("-" * 80)

from nltk import bigrams, trigrams

text_tokens = word_tokenize("Natural language processing is a subfield of artificial intelligence")
bigram_list = list(bigrams(text_tokens))
trigram_list = list(trigrams(text_tokens))

print("Bigrams (2-word sequences):")
for bg in bigram_list[:5]:
    print(f"  {bg}")

print("\nTrigrams (3-word sequences):")
for tg in trigram_list[:5]:
    print(f"  {tg}")

# 10. TEXT SIMILARITY (Jaccard Similarity)
print("\n10. TEXT SIMILARITY (Jaccard)")
print("-" * 80)

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union) if union else 0

text1 = "Machine learning is a subset of artificial intelligence"
text2 = "Artificial intelligence includes machine learning"
text3 = "Python is a programming language"

similarity_1_2 = jaccard_similarity(text1, text2)
similarity_1_3 = jaccard_similarity(text1, text3)

print(f"Text 1: {text1}")
print(f"Text 2: {text2}")
print(f"Text 3: {text3}")
print(f"\nSimilarity (Text1 vs Text2): {similarity_1_2:.3f}")
print(f"Similarity (Text1 vs Text3): {similarity_1_3:.3f}")

print("\n" + "=" * 80)
print("NLTK EXAMPLES COMPLETED")
print("=" * 80)
