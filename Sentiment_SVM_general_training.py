# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:03:28 2019

@author: Country Dragon
"""
#Basic Libraries
import pandas as pd
import numpy as np

#Cleaning Libraries
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#BoW & TFIDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Convert All Agree & 75% Agree data to dataframe
df_all_agree = pd.read_csv("PhraseBank/Sentences_AllAgree.txt", sep="@", encoding="ISO-8859-1", header=None, names=['sentence','sentiment'])
df_75_agree = pd.read_csv("PhraseBank/Sentences_75Agree.txt", sep="@", encoding="ISO-8859-1", header=None, names=['sentence','sentiment'])

#Remove neutral statements
df_posneg_all = df_all_agree[df_all_agree.sentiment != 'neutral']
df_posneg_75 = df_75_agree[df_75_agree.sentiment != 'neutral']


#Combination of both produces duplicates but put emphasis on sentences with strong agreement (under 100% category)
df_all = pd.concat([df_posneg_all,df_posneg_75],ignore_index=1)

df_all.dtypes
#Cleaning text class
class CleanText(BaseEstimator, TransformerMixin):
    def remove_currency(self, input_text): # Effective replacement code for character
        currency_list = ['EUR','USD','€','$']
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
        clean_X = X.apply(self.remove_currency).apply(self.remove_symbols).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.correct_million).apply(self.correct_billion).apply(self.to_lower).apply(self.remove_stopwords)#.apply(self.stemming)
        return clean_X
    
ct = CleanText()
df_cleaned = ct.fit_transform(df_all['sentence'])

#TFIDF feature selection

# TF-IDF features (Must not occur in more than 70% of docs and minimum must appear in 2 docs)
tfidf_word_vectorizer = TfidfVectorizer(max_df=0.70, min_df=2, ngram_range = (1,2), stop_words='english') 

# TF-IDF feature matrix
tfidf_word_feature = tfidf_word_vectorizer.fit_transform(df_cleaned)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer = tfidf_word_feature[0]

 
# place tf-idf values in a pandas data frame
tfidf_list = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_word_vectorizer.get_feature_names(), columns=["tfidf"])
tfidf_list = tfidf_list.sort_values(by=["tfidf"],ascending=False)


#Model Training
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn import svm

target_variable = df_all['sentiment'].apply(lambda x: 0 if x=='negative' else 1 )

#Splitting the master set to training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(tfidf_word_feature, target_variable, test_size=0.2, random_state=870)


#do a 10-fold cross validation on training set (representing both training and CV set together)
kf = KFold(n_splits = 10)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

scores = ['precision', 'recall','f1','accuracy']
#scores =['accuracy']

#Finding the optimal setting based on 4 criteria: precision, recall, f1 and accuracy
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=kf,
                       #scoring='%s_macro' % score)
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


#Retrain the model using the optimum settings after considering all 4 criteria
    
clf = svm.SVC(kernel = 'linear', C=50)
clf.fit(X_train, y_train)

#Predicting the test set and evaluating the results
from sklearn import metrics

y_pred = clf.predict(X_test)

print("Confusion Matrix: \n",metrics.confusion_matrix(y_test,y_pred))
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#Saving the model
import joblib

joblib_file = "SVM_model.pkl"
joblib.dump(clf,joblib_file)


#Loading the model
joblib_file = "SVM_model.pkl"
clf = joblib.load(joblib_file)


#Testing on new data
text = ['The KLCI surged and closed at 1,597.98 points amid overnight gains in US market after the Federal Reserve cut interest rates for the third time this year']


df_text = pd.DataFrame(text)
cleaned_text = ct.fit_transform(df_text[0])
new_test_term_TFIDF = tfidf_word_vectorizer.transform(cleaned_text)

clf.predict(new_test_term_TFIDF)