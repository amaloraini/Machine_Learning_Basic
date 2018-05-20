from nltk import ne_chunk, word_tokenize, pos_tag
import nltk

#You might have to download the following
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger')

def POS_tagging(text):
	print(ne_chunk(pos_tag(word_tokenize(text))))

#interesting api check ambiverse.com

POS_tagging("I love to drive BMW")