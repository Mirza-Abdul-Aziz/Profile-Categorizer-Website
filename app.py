import os
import re
import pandas as pd
import numpy as np

import pickle
from nltk import TweetTokenizer
from nltk import PorterStemmer
from scipy import rand
import scipy.sparse as sp

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from flask import Flask, render_template, redirect, url_for, request, session, make_response
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
history_s = {}
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/delSession')
def delSession():
	session.clear()
	return redirect(url_for('history'))

@app.route('/history')
def history():
	return render_template("history.html", session = session)

@app.route('/search', methods=["POST", "GET"]) 
def search():
    return render_template("search.html")

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		username = request.form['user_url']
	if username.startswith('@'):
		os.system("snscrape --jsonl --max-results 10 twitter-search 'from:{}'> user-tweets.json".format(username))
	tweets_df = pd.read_json('user-tweets.json', lines=True)
	tweets_df.to_csv(f'./raw_tweets_data.csv', mode='w')
	cleaned_df = pd.read_csv("./raw_tweets_data.csv", skipinitialspace=True)
	cleaned_df.columns = cleaned_df.columns.str.rstrip()

	scraped_headers = list(cleaned_df)

	save_columns = ['content', 'user'] # columns NOT to be dropped  

	# cleaning dataset, dropping extra columns

	for heading in scraped_headers:
		if heading not in save_columns:
			cleaned_df.drop(heading, axis=1, inplace=True) 

	rows = cleaned_df['user']

	i = 0
	for row in rows:
		# converting to dictionary
		row_dict = ast.literal_eval(row) 
		cleaned_dict = {'url': row_dict['url'], 'description': row_dict['description'], 'verified': row_dict['verified'], 'tweets': cleaned_df.loc[i,'content'], 'location': row_dict['location']}
		temp_df = pd.DataFrame(cleaned_dict, index=[0])
		if i != 0:
			temp_df.to_csv('./cleaned_dataset.csv', mode='a', index=False, header=None)
		else:
			temp_df.to_csv('./cleaned_dataset.csv', mode='w', index=False)
		i = i + 1
	content_df = pd.read_csv("./cleaned_dataset.csv", skipinitialspace=True)

	i = 0

	row_count = len(content_df.index)

	for j in range(row_count-1):
		
		tweets = ''
		
		if(i >= row_count-1):
			break
		
		while(content_df.loc[i,'url'] == content_df.loc[i+1,'url']):

			tweets += content_df.loc[i,'tweets']
				
			content_df.drop([i], inplace = True)   

			i = i+1

			if(i == row_count-1):
				tweets += content_df.loc[i,'tweets']
				break

				
		content_df.loc[i,'tweets'] = tweets
		i = i+1
	

	content_df.to_csv('./final_dataset.csv', mode='w', index=None)
	#Reading correct dataset
	sample_data=pd.read_csv('final_dataset.csv',encoding='latin1')
	sample_data.drop(sample_data.columns[sample_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
	sample_data = sample_data.drop(['url', 'verified', 'location'], axis=1)
	columns = ['description', 'tweets']
	for column in columns:
		def remove_pattern(column_data, pattern):
			processed_data = re.sub(pattern,"", column_data)
			return processed_data

		#Removing twitter handles
		sample_data["Processed "+column] = np.vectorize(remove_pattern)(sample_data[column], "@[\w]*")
		
		#removing punctuations
		sample_data["Processed "+column] = sample_data["Processed "+column].str.replace("[^a-zA-Z#\s]", "")

		#Removing short words
		sample_data["Processed "+column] = sample_data["Processed "+column].apply(
		lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

		#Tokenization
		Tokenizer = TweetTokenizer()
		sample_data["Processed "+column] = sample_data["Processed "+column].apply(lambda x: Tokenizer.tokenize(str(x)))

		#Stemming
		ps = PorterStemmer()
		sample_data["Processed "+column] = sample_data["Processed "+column].apply(lambda string: [ps.stem(letter) for letter in string])
		
		#Stiching the tokens back
		for i in range(len(sample_data["Processed "+column])):
			sample_data["Processed "+column][i] = ' '.join(sample_data["Processed "+column][i])
	X = sample_data.iloc[:, 2:]
	description_tfidf_vectorizer = pickle.load(open('description_tfidf_vectorizer.pkl', 'rb'))
	tweets_tfidf_vectorizer = pickle.load(open('tweets_tfidf_vectorizer.pkl', 'rb'))
	description_tfidf_vectors = description_tfidf_vectorizer.transform(X['Processed description'])
	tweets_tfidf_vectors = tweets_tfidf_vectorizer.transform(X['Processed tweets'])
	combined_user_input = sp.hstack([description_tfidf_vectors, tweets_tfidf_vectors], format='csr')
	combined_user_input.todense()
	model = pickle.load(open('lr_tfidf_trained_model.pkl', 'rb'))
	prediction = model.predict(combined_user_input)
	key = time.ctime()
	if prediction == 0:
		session[str(key)] = [username, "Actor"]
		return render_template("prediction.html", prediction = "Actor")
	elif prediction == 1:
		session[str(key)] = [username, "Content Creator"]
		return render_template("prediction.html", prediction = "Content CReator")
	elif prediction == 2:
		session[str(key)] = [username, "Education"]
		return render_template("prediction.html", prediction = "Education")
	elif prediction == 3:
		session[str(key)] = [username, "Politician"]
		return render_template("prediction.html", prediction = "Politician")
	elif prediction == 4:
		session[str(key)] = [username, "Singer"]
		return render_template("prediction.html", prediction = "Singer")
	elif prediction == 5:
		session[str(key)] = [username, "Sports"]
		return render_template("prediction.html", prediction = "Sports")
if __name__ == "__main__":
    app.run(debug = True)