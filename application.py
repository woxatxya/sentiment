# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:06:30 2020

@author: 54425
"""

import re
import string
#import random
import nltk
from nltk.tag import pos_tag, pos_tag_sents
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
#from nltk import FreqDist

import numpy as np
import json
import pickle

import flask
from flask import request
from flask import Flask

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

application = Flask(__name__)

def cleaned_list_func(evert_tweet):
    new_text = []
    cixing_list = pos_tag(evert_tweet)
    for word, cixing in cixing_list:
        word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:[0-9a-fA-F][0-9a-fA-F]))+', '', word)
        word = re.sub('(@[A-Za-z0-9_]+)', '', word)
        if cixing.startswith('NN'):
            pos = 'n'
        elif cixing.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        new_word = lemmatizer.lemmatize(word, pos)
        if len(new_word) > 0 and new_word not in string.punctuation and new_word.lower() not in stopwords.words('english'):
            new_text.append(new_word.lower())
    return new_text

def test(model, test_text):
    """
    :param model:
    :param test_text: 
    :return:
    """
    from nltk.tokenize import word_tokenize

    custom_tokens = cleaned_list_func(word_tokenize(test_text))
    result = dict([token, True] for token in custom_tokens)
    yuce_res = model.classify(result)
    #print('content: {} prediction: {}'.format(test_text, yuce_res))
    return yuce_res

def clean_tweets(raw_tweets):
    tweets = []
    for num_city in range(len(raw_tweets)):
        city = raw_tweets[num_city]['city_name']
        for tweet_num in range(len(raw_tweets[0]['tweets'])):
            tweets.append((city,raw_tweets[0]['tweets'][tweet_num]['text']))  
    return tweets    

def generate_json(tweets,sentiments):
    contents = []
    for i in range(len(tweets)):
        contents.append({'city':tweets[i][0],
         'tweet':tweets[i][1],
         'sentiment':sentiments[i]})
    contents = json.dumps(contents)
    return contents
'''
@application.route('/',methods=['GET','POST'])
def say_hello():
    return "success"

'''
@application.route('/',methods=['GET','POST'])
def say_hello():
    f = open('my_classifier.pickle', 'rb')
    model = pickle.load(f)
    f.close()
    
    raw_tweets =request.get_json()
    tweets = clean_tweets(raw_tweets)
    sentiments = []
    
    for tweet in tweets:
        print(tweet[1])
        sentiments.append(test(model, tweet[1]))
        
    tweets = generate_json(tweets,sentiments)
    
    return tweets

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    #application.debug = True
    application.run()