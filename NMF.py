"Non-negative Matrix multiplication"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.sparse import csr_matrix
import pandas as pd 
# LDA is based on probabilistic graphical modeling while NMF relies on linear algebra. 
#LDA = Latent Dirichlet Allocation 
https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
'''
tfidf toy example
documents = ["I love to eat pizza", "I like to eat hamburger"]
tfidf = TfidfVectorizer() 
csr_mat = tfidf.fit_transform(documents)
# Print result of toarray() method
print(csr_mat.toarray())
# Get the words: words
words = tfidf.get_feature_names()
# Print words
print(words)
'''
#tfidf = TfidfVectorizer() 
#csr_mat = tfidf.fit_transform(documents)
df = pd.read_csv("Datasets/wiki_articles/wikipedia-vectors.csv", index_col=0)
words = df.transpose().columns 
print (words)

print(df.shape)
articles = csr_matrix(df.transpose())
print(articles.shape)
titles = list(df.columns)
print (titles)
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
dfs = pd.DataFrame(nmf_features, index=titles)
#feature 3 is high 
print(dfs.loc['Anne Hathaway'])
print(dfs.loc['Denzel Washington'])

components_df = pd.DataFrame(model.components_, columns=words)
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())

