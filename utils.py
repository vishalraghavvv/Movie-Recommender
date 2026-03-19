"""
Shared utility functions for the movie recommender system
Used by both app.py and MovieRecommender.ipynb
"""

import ast
from nltk.stem.porter import PorterStemmer

def convert(obj):
    """Extract names from JSON-like string representation"""
    L = []
    data = obj if isinstance(obj, list) else ast.literal_eval(obj)
    for i in data:
        if isinstance(i, dict):
            L.append(i.get('name', ''))
        else:
            L.append(i)
    return L

def convert3(obj):
    """Extract top 3 cast members"""
    L = []
    data = obj if isinstance(obj, list) else ast.literal_eval(obj)
    counter = 0
    for i in data:
        if counter < 3:
            if isinstance(i, dict):
                L.append(i.get('name', ''))
            else:
                L.append(i)
            counter += 1
        else:
            break
    return L

def director_name(obj):
    """Extract director names from crew data"""
    L = []
    data = obj if isinstance(obj, list) else ast.literal_eval(obj)
    for i in data:
        if isinstance(i, dict):
            if i.get('job') == 'Director':
                L.append(i.get('name', ''))
        else:
            L.append(i)
    return L

def stemmer(text):
    """Apply Porter Stemmer to text"""
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
