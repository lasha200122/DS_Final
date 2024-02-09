import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stop_words(tokens):
    filtered_tokens = [word for word in tokens if not word in stop_words]
    return filtered_tokens

def stem_tokens(tokens):
    stemmed_tokens = [ps.stem(word) for word in tokens]
    return stemmed_tokens

def lemmatize_tokens(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens

def preprocess_text(text, use_stemming=True):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

def vectorize_texts(texts, vectorizer_type='count'):
    """
    Vectorize a list of preprocessed texts.
    """
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer()
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid vectorizer type. Choose 'count' or 'tfidf'.")

    vectorized_texts = vectorizer.fit_transform(texts)
    return vectorized_texts, vectorizer
