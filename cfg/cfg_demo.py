import nltk

from nltk import CFG
from nltk.parse.generate import generate

# https://www.nltk.org/howto/parse.html
# https://www.nltk.org/howto/grammar.html

grammar = nltk.CFG.fromstring("""
  S -> NP VP
  NP -> Det N | N
  VP -> V NP
  VP -> Vintransitive P NP
  Vintransitive -> 'sat'
  Det -> 'a' | 'the'
  N -> 'dog' | 'cat' | 'alpaca' | 'bat'
  V -> 'chased' | 'serenaded'
  P -> 'on' | 'in'
""")


for sent in nltk.parse.generate.generate(grammar):
  print(sent)
