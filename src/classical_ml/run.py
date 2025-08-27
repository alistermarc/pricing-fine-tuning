import pickle
import json
import math
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec
# from gensim.utils import simple_preprocess
from src.tester import Tester

# Load data
with open("data/train.pkl", "rb") as f:
    train = pickle.load(f)
with open("data/test.pkl", "rb") as f:
    test = pickle.load(f)

# Create a new "features" field on items, and populate it with json parsed from the details dict
for item in train:
    item.features = json.loads(item.details)
for item in test:
    item.features = json.loads(item.details)

# Feature engineering functions
def get_weight(item):
    weight_str = item.features.get('Item Weight')
    if weight_str:
        parts = weight_str.split(' ')
        try:
            amount = float(parts[0])
            unit = parts[1].lower()
            if unit=="pounds":
                return amount
            elif unit=="ounces":
                return amount / 16
            elif unit=="grams":
                return amount / 453.592
            elif unit=="milligrams":
                return amount / 453592
            elif unit=="kilograms":
                return amount / 0.453592
            elif unit=="hundredths" and parts[2].lower()=="pounds":
                return amount / 100
        except (ValueError, IndexError):
            return None
    return None

weights = [get_weight(t) for t in train]
weights = [w for w in weights if w]
_average_weight = sum(weights)/len(weights) if weights else 0

def get_weight_with_default(item):
    weight = get_weight(item)
    return weight or _average_weight

def get_rank(item):
    rank_dict = item.features.get("Best Sellers Rank")
    if rank_dict:
        try:
            if isinstance(rank_dict, dict):
                ranks = rank_dict.values()
                return sum(ranks)/len(ranks)
            elif isinstance(rank_dict, list):
                # handle cases where rank is a list of strings with numbers
                total_rank = 0
                count = 0
                for rank_str in rank_dict:
                    # extract numbers from string
                    nums = [int(s) for s in rank_str.split() if s.isdigit()]
                    if nums:
                        total_rank += sum(nums)
                        count += len(nums)
                return total_rank / count if count > 0 else 0
        except (AttributeError, TypeError, ValueError):
            return 0
    return 0

ranks = [get_rank(t) for t in train]
ranks = [r for r in ranks if r]
_average_rank = sum(ranks)/len(ranks) if ranks else 0

def get_rank_with_default(item):
    rank = get_rank(item)
    return rank or _average_rank

def get_text_length(item):
    return len(item.test_prompt())

TOP_ELECTRONICS_BRANDS = ["hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]
def is_top_electronics_brand(item):
    brand = item.features.get("Brand")
    return brand and brand.lower() in TOP_ELECTRONICS_BRANDS

def get_features(item):
    return {
        "weight": get_weight_with_default(item),
        "rank": get_rank_with_default(item),
        "text_length": get_text_length(item),
        "is_top_electronics_brand": 1 if is_top_electronics_brand(item) else 0
    }

# A utility function to convert our features into a pandas dataframe
def list_to_dataframe(items):
    features = [get_features(item) for item in items]
    df = pd.DataFrame(features)
    df['price'] = [item.price for item in items]
    return df

train_df = list_to_dataframe(train)
test_df = list_to_dataframe(test)

# Traditional Linear Regression!
np.random.seed(42)

# Separate features and target
feature_columns = ['weight', 'rank', 'text_length', 'is_top_electronics_brand']

X_train = train_df[feature_columns]
y_train = train_df['price']
X_test = test_df[feature_columns]
y_test = test_df['price']

# Train a Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict price for a new item
def linear_regression_pricer(item):
    features = get_features(item)
    features_df = pd.DataFrame([features])
    return model.predict(features_df)[0]

Tester.test(linear_regression_pricer)

# For the next few models, we prepare our documents and prices
# Note that we use the test prompt for the documents, otherwise we'll reveal the answer!!
prices = np.array([float(item.price) for item in train])
documents = [item.test_prompt() for item in train]

# Use the CountVectorizer for a Bag of Words model
np.random.seed(42)
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)
regressor = LinearRegression()
regressor.fit(X, prices)

def bow_lr_pricer(item):
    x = vectorizer.transform([item.test_prompt()])
    return max(regressor.predict(x)[0], 0)

Tester.test(bow_lr_pricer)

# # The amazing word2vec model, implemented in gensim NLP library
# np.random.seed(42)

# # Preprocess the documents
# processed_docs = [simple_preprocess(doc) for doc in documents]

# # Train Word2Vec model
# w2v_model = Word2Vec(sentences=processed_docs, vector_size=400, window=5, min_count=1, workers=8)

# # This step of averaging vectors across the document is a weakness in our approach
# def document_vector(doc):
#     doc_words = simple_preprocess(doc)
#     word_vectors = [w2v_model.wv[word] for word in doc_words if word in w2v_model.wv]
#     return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(w2v_model.vector_size)

# # Create feature matrix
# X_w2v = np.array([document_vector(doc) for doc in documents])

# # Run Linear Regression on word2vec
# word2vec_lr_regressor = LinearRegression()
# word2vec_lr_regressor.fit(X_w2v, prices)

# def word2vec_lr_pricer(item):
#     doc = item.test_prompt()
#     doc_vector = document_vector(doc)
#     return max(0, word2vec_lr_regressor.predict([doc_vector])[0])

# Tester.test(word2vec_lr_pricer)

# Support Vector Machines
np.random.seed(42)
svr_regressor = LinearSVR()
svr_regressor.fit(X_w2v, prices)

def svr_pricer(item):
    np.random.seed(42)
    doc = item.test_prompt()
    doc_vector = document_vector(doc)
    return max(float(svr_regressor.predict([doc_vector])[0]),0)

Tester.test(svr_pricer)

