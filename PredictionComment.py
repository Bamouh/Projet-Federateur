
# coding: utf-8

# In[3]:


import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


# In[4]:


#function definition
def clean_tweet(tweet):#function used to clean a single tweet
    tweet = str(tweet)
    tweet = re.sub("(http:\/\/)[^ ]*","",tweet)
    tweet = tweet.replace("&amp;","")
    tweet = re.sub("[^\w\s]","",tweet)
    tweet = re.sub("[\d]","",tweet)
    tweet = tweet.lower()
    return tweet

def comment_prediction(model, comment, count_vect, tfidf_transformer):
    test_data_counts = count_vect.transform([clean_tweet(comment)])
    test_data_tfidf = tfidf_transformer.transform(test_data_counts)
    predicted = model.predict(test_data_tfidf)
    return predicted


# In[7]:


#LOAD MODELS
model = joblib.load('finalized_model.sav')
count_vect = joblib.load('finalized_countvectorizer.sav')
tfidf_transformer = joblib.load('finalized_tfidftransformer.sav')

#PREDICTION DE TOXICITE DE COMMENTAIRE
toxicite = comment_prediction(model, "fuck off", count_vect, tfidf_transformer)
if toxicite == 0 :
    print("Commentaire non toxique")
elif toxicite == 1:
    print("Commentaire toxique")
elif toxicite == 2:
    print("Commentaire insultant")

