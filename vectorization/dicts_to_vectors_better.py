#!/usr/bin/env python3

import nltk

import sklearn


# documentation at
# https://scikit-learn.org/stable/modules/generated/
#   sklearn.feature_extraction.DictVectorizer.html

training_set = [
  ({"cat": 16, "rabbit": 2}, "cat"),
  ({"cat": 0, "rabbit": 1}, "rabbit")
  ]

vectorizer = sklearn.feature_extraction.DictVectorizer()

# set up the dictionary
featuresets = [fs for (fs,label) in training_set]
vectorizer.fit(featuresets)

vec = vectorizer.transform({"cat": 100, "rabbit": 1})
print(vec)


# vec = vectorizer.transform({"cat": 100, "rabbit": 1, "alpaca": 15})
