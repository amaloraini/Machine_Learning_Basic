from nltk.corpus import wordnet as wn

#http://www.nltk.org/howto/wordnet.html


print(wn.synsets('wolf')) #the POS VERB, NOUN, ADJ and ADV
print(wn.synset('wolf.n.01').definition())

dog = wn.synset('dog.n.01')
wolf = wn.synset('wolf.n.01')
cat = wn.synset('cat.n.01')
hit = wn.synset('hit.v.01')
slap = wn.synset('slap.v.01')

print(dog.hyponyms())
print(dog.member_holonyms())

print(dog.path_similarity(cat))
print(dog.path_similarity(wolf))
