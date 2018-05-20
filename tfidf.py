from sklearn.feature_extraction.text import TfidfVectorizer

tokenize = lambda doc: doc.lower().split(" ")
lectures = ["this is some food", "this is some drink"]
vectorizer = TfidfVectorizer(ngram_range=(1,2), norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, stop_words='english')

X = vectorizer.fit_transform(lectures)
features_by_gram = []
for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    features_by_gram.append((len(f.split(' ')), w, f))
top_n = 2
features_by_gram.sort(key=lambda x: x[1])
for z in features_by_gram:
	print(z)
