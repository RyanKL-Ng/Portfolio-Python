# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:07:45 2019

@author: Country Dragon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests

from urllib.request import urlopen
from bs4 import BeautifulSoup as bs


def remove_symbols(input_text):
    return re.sub(r'[^\w.|]', ' ', input_text)

def remove_digits(input_text):
    return re.sub('\d*\.\d+', '', input_text)

def cleanText(inputText):
    
    url = inputText
    req = urlopen(url)
    raw = req.read()
    soup = bs(raw)
    
    #Get main text body
    main_data = soup.find('div', {'class': 'col-md-10 col-sm-10 col-xs-10 artbody'})

    #Remove all links from the body
    if type(main_data.find_all('a')) is not None:
            links = main_data.find_all('a')
            for link in links:        
                link.extract()

    #Remove the mini title from the body (else the mini title will join with the first paragraph)
    h2_data = main_data.find('h2')
    h2_data.extract()
    
    #Extract and clean the main text
    text = main_data.get_text('.', strip=True)
    new_text = remove_symbols(text)
    new_text = remove_digits(new_text)
    cleaned_text = " ".join(new_text.split())
    
    #Split the main text into rows and place inside a Dataframe
    df =cleaned_text.split('.')
    df.insert(0,h2_data.get_text())
    df_list = pd.DataFrame(df)
    df_filtered_list = df_list[df_list[0].str.len() > 15]
    df_filtered_list = df_filtered_list.reset_index(drop=True)
    
    return df_filtered_list[0].apply(lambda x: x.strip())



url = 'https://www.forexlive.com/centralbanks'

html = urlopen(url)
soup = bs(html,'lxml')

urllist = []
urllist.append('https://www.forexlive.com/centralbanks')
urlheader = 'https://www.forexlive.com/centralbank/Headlines/'

for x in range(1,10):
    urlAdd = urlheader+str(x)
    urllist.append(urlAdd)

mylist = []

for row in urllist:
    htmlrow = urlopen(row)
    souprow = bs(htmlrow,'lxml')    
    for link in souprow.find_all(href=re.compile("forexlive.com/centralbank/!/")):
        print(link.get('href'))
        mylist.append(link.get('href'))
        
    
    
mylist = list(dict.fromkeys(mylist))


df = pd.DataFrame(mylist)

fixedlist = []

for row in df[0]:
    if row[0:1] == '/':    
        newdata = 'https:'+row
        fixedlist.append(newdata)
    else:
        newdata = row
        fixedlist.append(newdata)

df['sites'] = fixedlist
df = df.drop(0,axis=1)

df.to_csv('forexlive_test_sites.csv')

sentences = []

for row in df['sites']:
    extracted_text = cleanText(row)
    
    for text in extracted_text:        
        sentences.append(text)

sentences = pd.DataFrame(sentences)
sentences.to_csv('forexlive.csv')

