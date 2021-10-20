#!/usr/bin/env python3

import nltk

training_set = [
  ({"cat": 16, "rabbit": 2}, "cat"),
  ({"cat": 0, "rabbit": 1}, "rabbit")
  ]


classifier = nltk.classify.MaxentClassifier.train(training_set, max_iter=10)

print(classifier.classify({"cat": 16, "rabbit": 2}))

print(classifier.classify({"cat": 100, "rabbit": 1}))

print(classifier._encoding.encode({"cat": 16, "rabbit": 2}, "cat"))
print(classifier._encoding.encode({"cat": 15, "rabbit": 2}, "cat"))

