#Cosine Similarity 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
doc1 = np.array([1, 0, -1])
doc2 = np.array([-1,-1, 0])

print(doc1)
print(doc2)
doc1 = doc1.reshape(1, -1)
doc2 = doc2.reshape(1, -1)

print(doc1)
print(doc2)

print((1-cosine_similarity(doc1, doc2)))
