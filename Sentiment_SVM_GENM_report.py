# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:23:43 2019

@author: Country Dragon
"""
import joblib
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from tika import parser
from bs4 import BeautifulSoup
from io import StringIO
import os

#BoW & TFIDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class CleanText(BaseEstimator, TransformerMixin):
    def replace_ringgit(self, input_text): # Effective replacement code for character
        currency_list = ['RM','Ringgit','ringgit']
        regex = re.compile('|'.join(map(re.escape,currency_list)))
        return regex.sub('ringgitCurrency ', input_text)
    
    def replace_dollar(self, input_text): # Effective replacement code for character
        currency_list = ['USD','$','dollar','Dollar']
        regex = re.compile('|'.join(map(re.escape,currency_list)))
        return regex.sub('dollarCurrency ', input_text)    

    def replace_currency(self, input_text): # Effective replacement code for character
        currency_list = ['Euro','euro','€','Yen','yen','¥','Pound','British Pound','GBP','£']
        regex = re.compile('|'.join(map(re.escape,currency_list)))
        return regex.sub('currencySign ', input_text)    
    
    def remove_symbols(self, input_text):
        return re.sub(r'[^\w]', ' ', input_text)   
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation removed
        return input_text.translate(trantab)
    
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)    

    def correct_million(self, input_text):
        replace_list = ['m','mln','mn'] # Effective replacement code for words
        regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape,replace_list)))
        return regex.sub('million', input_text)
    
    def correct_billion(self, input_text):
        replace_list = ['b','bln','bn']
        regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape,replace_list)))
        return regex.sub('billion', input_text)    
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = []
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.replace_ringgit).apply(self.replace_dollar).apply(self.replace_currency).apply(self.remove_symbols).apply(self.remove_digits).apply(self.remove_punctuation).apply(self.to_lower).apply(self.remove_stopwords)
        return clean_X



def remove_symbols(self, input_text):
        return re.sub(r'[^\w]', ' ', input_text)

def remove_digits(self, input_text):
    return re.sub('\d*\.\d+', '', input_text) 

#Loading model
joblib_file = "SVM_model_genm_announcement.pkl"
clf = joblib.load(joblib_file)

joblib_tfidf_file = "tfidf_dict_genm_announcement.pkl"
tfidf_word_feature = joblib.load(joblib_tfidf_file)

joblib_tfidf_vec_file = "tfidf_genm_vectorizer.pkl"
tfidf_word_vectorizer = joblib.load(joblib_tfidf_vec_file)

#Loading the text cleaner
ct = CleanText()


_buffer = StringIO()
parsername = 'More Original PDF Announcement/19Q3.pdf'#'PDF Processing/19Q2.pdf'
file_data = []


xml = parser.from_file(parsername, xmlContent = True)
xmltext = xml['content']
beautifiedText = BeautifulSoup(xmltext)

for page, content in enumerate(beautifiedText.find_all('div', attrs={'class': 'page'})):
    #print('Parsing page {} of pdf file...'.format(page+1))
    _buffer.write(str(content))
    parsed_content = parser.from_buffer(_buffer.getvalue())
    _buffer.truncate()
    file_data.append({'id': 'page_'+str(page+1), 'content': parsed_content['content']})

main_text = file_data[len(file_data)-1]['content']
firstRoundCleaning = remove_digits(main_text, main_text)
firstRoundCleaning = " ".join(firstRoundCleaning.split())
separate_text = firstRoundCleaning.split('.')
separate_text = separate_text[:len(separate_text)-1]

df_text = pd.DataFrame(separate_text)
cleaned_text = ct.fit_transform(df_text[0])
new_test_term_TFIDF = tfidf_word_vectorizer.transform(cleaned_text)

df_text['sentiment'] = clf.predict(new_test_term_TFIDF)

positive_sentiment_score = df_text['sentiment'].value_counts(1)[1]
negative_sentiment_score = df_text['sentiment'].value_counts(1)[0]
document_sentiment_score = positive_sentiment_score - negative_sentiment_score