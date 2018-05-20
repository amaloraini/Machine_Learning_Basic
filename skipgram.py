from nltk.util import skipgrams

#from A Closer Look  at Skip - gram  Modelling (Guthrie et al.) paper

sent = "Insurgents killed in ongoing fighting".split()

print(sent)

two_skip_two_gram = list(skipgrams(sent, 2, 2))

# 2-skip bi-bigram 
print(two_skip_two_gram)

# 3-skip bi-gram
three_skip_bi_gram = list(skipgrams(sent, 2, 3))
print(three_skip_bi_gram)
# 2-skip trigram
two_skip_tri_gram = list(skipgrams(sent, 3, 2))
print(two_skip_tri_gram)