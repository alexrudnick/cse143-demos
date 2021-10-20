import nltk


def sentence_to_features(sent):
  out = nltk.FreqDist()
  tokens = nltk.word_tokenize(sent)
  # for c in sent:
  #   out[c] +=1
  for token in tokens:
    out[token] += 1
  return out

def interact():
  import code
  code.InteractiveConsole(locals=globals()).interact()

def build_classifier():
  training_set = []

  with open("europarl_en_1k.txt") as infile:
    for line in infile:
      line = line.strip()
      features = sentence_to_features(line)
      instance = (features, "en")
      training_set.append(instance)

  with open("europarl_es_1k.txt") as infile:
    for line in infile:
      line = line.strip()
      features = sentence_to_features(line)
      instance = (features, "es")
      training_set.append(instance)

  ## cutoffs = {"max_iter": 50}
  classifier = nltk.classify.MaxentClassifier.train(training_set)
  return classifier

def load_test_set():
  test_set = []

  with open("test_set_es") as infile:
    for line in infile:
      line = line.strip()
      features = sentence_to_features(line)
      instance = (features, "es")
      test_set.append(instance)

  with open("test_set_en") as infile:
    for line in infile:
      line = line.strip()
      features = sentence_to_features(line)
      instance = (features, "en")
      test_set.append(instance)
  return test_set


def main():
  classifier = build_classifier()

  test_set = load_test_set()

  acc = nltk.classify.accuracy(classifier, test_set)
  print("we got an accuracy of:", acc)

# instance = sentence_to_features("this is my sentence in English")

if __name__ == "__main__": main()
