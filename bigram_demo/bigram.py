
import nltk
import math
import random
import sys

from collections import defaultdict

def load_sentences(filename):
  """give us a list of sentences where each sentence is a list of tokens.
  Assumes the input file is one sentence per line, pre-tokenized."""
  out = []
  with open(filename) as infile:
      for line in infile:
          line = line.strip()
          tokens = line.split()
          out.append(tokens)
  return out


LOW_NUMBER = 0.00001
def counts_to_probs(counts_dict):
    """Take a counts dictionary and return the appropriate probs dictionary."""
    out = defaultdict(lambda: defaultdict(lambda: LOW_NUMBER))
    for prev in counts_dict:
        for cur in counts_dict[prev]:
            out[prev][cur] = (counts_dict[prev][cur] /
                              sum(counts_dict[prev].values()))
    return out

def score_sentence(sent, bigram_probs, unigram_probs):
    """Return the surprisal value of this sentence in bits."""
    total_surprise = 0

    sent = ["**START**"] + sent + ["**END**"]
    for (prev, cur) in zip(sent, sent[1:]):
        if prev in bigram_probs and cur in bigram_probs[prev]:
            surprise = -math.log(bigram_probs[prev][cur], 2)
        else:
            ## STUPID BACKOFF (Brants et al 2007)
            prob = 0.4 * unigram_probs[cur]
            surprise = -math.log(prob, 2)
        total_surprise += surprise
    return total_surprise

def main():
    words_to_freq_dists = defaultdict(nltk.FreqDist)

    sentences = load_sentences(sys.argv[1]) # "frankenstein-sentences.txt")

    for sentence in sentences:
      prev = "**START**"
      for word in (sentence + ["**END**"]):
        bigram = (prev, word)
        words_to_freq_dists[prev][word] += 1
        prev = word

    prev = "**START**"

    for i in range(100):
      pool = []
      for (word, count) in words_to_freq_dists[prev].items():
        for j in range(count):
          pool.append(word)
      nextword = random.choice(pool)
      if nextword == "**END**":
        break
      print(nextword, end=" ")
      prev = nextword
    print()

if __name__ == "__main__": main()
