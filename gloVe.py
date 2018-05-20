from gensim.models import KeyedVectors

pretrained_file = 'pretrained/glove/glove.6B.50d.txt'   #add the following [vec_size dimension_size] as the first line. For 50d, we have 400000 50

print('---Loading gloVe model---')
model = KeyedVectors.load_word2vec_format(pretrained_file, binary = False)

#remember all vocabulary is NOT capitalized and no two words queries such as 'new york'. 
print(model['Berlin']) #50d vectors

germany_capital = model.most_similar(positive=['china', 'paris'], negative = ['france'])
print(germany_capital)

#find the top similar words to toyota
most_similar_words = model.most_similar('toyota', topn=5)
print(most_similar_words)
'''
output of the 50d
[('berlin', 0.9203965663909912), ('frankfurt', 0.8201637268066406), ('vienna', 0.8182449340820312), ('munich', 0.8152028918266296), ('hamburg', 0.7986699342727661), ('stockholm', 0.7764843106269836), ('budapest', 0.7678730487823486), ('warsaw', 0.7668998837471008), ('prague', 0.7664733529090881), ('amsterdam', 0.7555989027023315)]
[('honda', 0.9040561318397522), ('automaker', 0.8352562785148621), ('nissan', 0.8328733444213867), ('bmw', 0.8130687475204468), ('auto', 0.8112044930458069)]
[Finished in 39.3s]

output of the 300d
[('berlin', 0.8127261400222778), ('frankfurt', 0.7258865833282471), ('munich', 0.6659741401672363), ('cologne', 0.6438040137290955), ('bonn', 0.6358616352081299), ('vienna', 0.6204482316970825), ('hamburg', 0.6023349761962891), ('leipzig', 0.5953505039215088), ('german', 0.5897991061210632), ('stuttgart', 0.5740550756454468)]
[('honda', 0.7437493801116943), ('nissan', 0.6815259456634521), ('automaker', 0.6738415956497192), ('camry', 0.653520941734314), ('prius', 0.6512324810028076)]
[Finished in 174.4s]
'''