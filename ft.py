from gensim.models.wrappers import FastText
import fasttext as ft

'''
.vec file: format is the same as .txt file format, and you could use it in other applications
.bin file: could be used if you want to continue training the vectors or to restart the optimization.
'''
'''
#CBOW model
model = ft.cbow('HEllo where are you', 'model')

#Skipgram
model = fasttext.skipgram(sentences, 'model')
'''
pretrained_file = 'pretrained/fasttext/wiki-news-300d-1M.vec'
model1 = ft.load_model(pretrained_file)
model2 = FastText.load_fasttext_format(pretrained_file)
print (model['machine']) # get the vector of the word 'machine'