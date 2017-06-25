import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer

def main():
	word_data = pickle.load( open('word_data.pkl', "r"))
	label = pickle.load( open('word_data_label.pkl', "r") )
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	features_train = vectorizer.fit_transform(word_data)
	clf = MultinomialNB().fit(features_train, label)
	pickle.dump(vectorizer,open("vectorizer.pkl","w"))
	pickle.dump(clf, open("text_learn_clf.pkl","w"))
	print 'finish'

def predictByEmail(email, _clf, vectorizer):
	from extract_email import getSingleWordData
	train_data = getSingleWordData(email)
	if not train_data:
		#empty list
		return 'NaN'
	else:
		feature = vectorizer.transform(train_data)
		pred =  _clf.predict(feature)
		return pred
if __name__ == "__main__":
	main()
