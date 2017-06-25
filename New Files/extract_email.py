import sys
import os
import pickle

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

def updateWordData(email,label,word_data,word_data_label):
	fn = "from_" + email + ".txt"
	try:
		from_file = open("./emails_by_address/" + fn, 'r')
    		for path in from_file:
			path = path[path.index('maildir'):]
       			path = os.path.join('..', path[:-1])
			email = open(path, 'r')
			words = parseOutText(email)
			word_data.append(words) 
			word_data_label.append(label)
			email.close()
		from_file.close()
	except IOError:
		pass

def getSingleWordData(email):
	fn = "from_" + email + ".txt"
	words = ""
	try:
		from_file = open("./emails_by_address/" + fn, 'r')
    		for path in from_file:
			path = path[path.index('maildir'):]
       			path = os.path.join('..', path[:-1])
			email = open(path, 'r')
			words = parseOutText(email)
			email.close()
		from_file.close()
	except IOError:
		return []
	return [words]

def main():
	data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
	data_dict.pop('TOTAL',0)

	word_data = []
	word_data_label = []

	for k,v in data_dict.iteritems():
		if v['email_address'] == 'NaN':
			pass
		else:
			updateWordData(v['email_address'],v['poi'],word_data,word_data_label)
			print len(word_data),len(word_data_label)


	pickle.dump( word_data, open("word_data.pkl", "w") )
	pickle.dump( word_data_label, open("word_data_label.pkl", "w") )

if __name__ == "__main__":
	main()
