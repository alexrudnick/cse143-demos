#/usr/bin/env python3

import nltk
import math

from nltk.corpus import brown

from collections import defaultdict
from collections import namedtuple

SMALLPROB = 1e-6

def learn_model():
  transitions = defaultdict(lambda: defaultdict(int))
  transition_probs = defaultdict(lambda: defaultdict(lambda: SMALLPROB))

  emissions = defaultdict(lambda: defaultdict(int))
  emission_probs = defaultdict(lambda: defaultdict(lambda: SMALLPROB))

  for tagged_sent in brown.tagged_sents():
    prev_tag = "<S>"
    for word,tag in tagged_sent:
      tag = nltk.tag.mapping.map_tag("brown", "universal", tag)
      transitions[prev_tag][tag] += 1
      emissions[tag][word] += 1
      emissions[tag]["*TOTAL*"] += 1
      prev_tag = tag

  for prevtag in transitions.keys():
    total = 0
    for tag in transitions[prevtag].keys():
      total += transitions[prevtag][tag]
    for tag in transitions[prevtag].keys():
      transition_probs[prevtag][tag] = transitions[prevtag][tag] / total

  for tag in emissions.keys():
    for word in emissions[tag].keys():
      if word == "*TOTAL*": continue
      emission_probs[tag][word] = (emissions[tag][word] / emissions[tag]["*TOTAL*"])

  return (transition_probs, emission_probs)


Cell = namedtuple('Cell', ['bestprev', 'score'])

def main():
  transition_probs, emission_probs = learn_model()

  possible_tags = list(emission_probs.keys())
  print(possible_tags)

  sentence = "wind it on the back .".split()
  # indexed by timestep, then inside there we have a dict from tag to cell.
  trellis = []
  for word in sentence:
    d = {}
    for tag in possible_tags:
      d[tag] = None # we will stick cells in here
    trellis.append(d)

  ## first timestep
  prev_tag = "<S>"
  for tag in possible_tags:
    score = math.log(transition_probs[prev_tag][tag], 2)
    score +=  math.log(emission_probs[tag][sentence[0]], 2)
    cell = Cell(bestprev=prev_tag, score=score)
    trellis[0][tag] = cell

  ## subsequent timesteps
  for timestep in range(1, len(sentence)):
    word = sentence[timestep]
    for tag in possible_tags:
      bestscore = float('-inf')
      bestprev = None
      emission = math.log(emission_probs[tag][word], 2)
      for prevtag in possible_tags:
        transition = math.log(transition_probs[prevtag][tag], 2)
        score = trellis[timestep-1][prevtag].score
        score += transition + emission

        if score > bestscore:
          bestscore = score
          bestprev = prevtag
      cell = Cell(bestprev=bestprev, score=bestscore)
      trellis[timestep][tag] = cell

  ## TRELLIS IS COMPLETE.
  max_final_score = float('-inf')
  max_final_tag = None
  final_timestep = len(sentence) - 1
  for tag in possible_tags:
    score = trellis[final_timestep][tag].score 
    if score > max_final_score:
      max_final_score = score
      max_final_tag = tag

  tags = [max_final_tag]
  tag = max_final_tag
  for timestep in range(len(sentence) - 1, -1, -1):
    tag = trellis[timestep][tag].bestprev
    tags.append(tag)
    # print("tag:", tag, "score:", trellis[1][tag])
  
  tags.reverse()
  for token,tag in zip(["(start)"] + sentence, tags):
    print(token, tag)

if __name__ == "__main__": main()
