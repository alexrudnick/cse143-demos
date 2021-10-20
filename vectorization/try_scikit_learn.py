#!/usr/bin/env python3

import nltk

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# sklearn.tree.DecisionTreeClassifier (max_depth = 5)

classifier = SklearnClassifier(LogisticRegression())

# classifier = SklearnClassifier(DecisionTreeClassifier(max_depth=5))

training_set = [
  ({"cat": 16, "rabbit": 2}, "cat"),
  ({"cat": 0, "rabbit": 10}, "rabbit")
  ]

classifier.train(training_set)

print(classifier.classify({"cat": 16, "rabbit": 2}))
print(classifier.classify({"cat": 15, "rabbit": 2}))
print(classifier.classify({"cat": 1, "rabbit": 100}))

