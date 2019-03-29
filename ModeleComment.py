
# coding: utf-8

# In[73]:


#importations
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import csv


# In[74]:


#function definition
def clean_tweet(tweet):#function used to clean a single tweet
    tweet = str(tweet)
    tweet = re.sub("(http:\/\/)[^ ]*","",tweet)
    tweet = tweet.replace("&amp;","")
    tweet = re.sub("[^\w\s]","",tweet)
    tweet = re.sub("[\d]","",tweet)
    tweet = tweet.lower()
    return tweet
def remove_common_and_rare_words(tweet,com,rar):#function used to remove the common words (in com) and rare words (in rar) from a single tweet
    tweetWords = tweet.split(" ")
    newTweet = ""
    for word in tweetWords:
        if (word not in com) & (word not in rar):
            newTweet += word + " "
    return newTweet
def remove_single_letter_words(tweet):#function used to remove single letter words ('m' 'n', 'I'...)
    tweetWords = tweet.split(" ")
    newTweet = ""
    for word in tweetWords:
        if len(word) != 1:
            newTweet += word + " "
    return newTweet
def clean_dataset(dataset):#function used to clean an entire dataset, uses the three previously declared functions
    dataset = dataset.apply(lambda x: clean_tweet(x))
    frequencyOfWords = pd.Series(' '.join(dataset).split()).value_counts()
    commonWords = frequencyOfWords[:10] # 10 most common words
    rareWords = frequencyOfWords[-10:] # 10 rarest words
    dataset = dataset.apply(lambda x: remove_common_and_rare_words(x,commonWords,rareWords))
    dataset = dataset.apply(lambda x: remove_single_letter_words(x))
    return dataset


# In[75]:


#load training set and testing set
train_data = pd.read_csv('train_preprocessed.csv')
test_data = pd.read_csv('test_preprocessed.csv')


# In[76]:


#Cleaning both training and testing sets
train_data['comment_text'] = clean_dataset(train_data['comment_text'])
test_data['comment_text'] = clean_dataset(test_data['comment_text'])


# In[77]:


#Apply features extraction algorithm to train set
count_vect = CountVectorizer(ngram_range=(1,1))
train_data_counts = count_vect.fit_transform(train_data['comment_text'])


# In[78]:


#Apply TF-IDF to training set
tfidf_transformer = TfidfTransformer()
train_data_tfidf = tfidf_transformer.fit_transform(train_data_counts)


# In[79]:


#Feed the cleaned tweets and their classification to the classifier
#print(train_data_tfidf.shape)
#print(train_data['toxic'].shape)
Y = train_data['insult'] + train_data['threat'] + train_data['toxic']
clf = MultinomialNB().fit(train_data_tfidf, Y)


# In[80]:


#Apply features extraction algorithm and TF-IDF to the testing set
test_data_counts = count_vect.transform(test_data['comment_text'])
test_data_tfidf = tfidf_transformer.transform(test_data_counts)


# In[81]:


#Predict the classification of the testing set
predicted = clf.predict(test_data_tfidf)


# In[83]:


#Display both tweets from the test set and their predicted classification
for tweet, category in zip(test_data['comment_text'], predicted):
        print('%r => %s' % (tweet, category))


# In[ ]:


#Create a csv file to store the submission
header = ["comment_text","toxic"]
rows = zip(test_data['comment_text'],predicted)
with open('sample_submission.csv', 'w') as submission:
    wr = csv.writer(submission, delimiter=',',lineterminator='\n', quoting=csv.QUOTE_ALL)
    wr.writerow(header)
    for row in rows:
        wr.writerow(row)


# In[93]:


from sklearn.externals import joblib
# now you can save it to a file
#joblib.dump(clf, 'filename.pkl') 
# and later you can load it
#clf = joblib.load('filename.pkl')

filename = 'finalized_model.sav'
joblib.dump(clf, filename)

filename = 'finalized_countvectorizer.sav'
joblib.dump(count_vect, filename)

filename = 'finalized_tfidftransformer.sav'
joblib.dump(tfidf_transformer, filename)


# In[91]:


test_data_counts = count_vect.transform([clean_tweet("f")])
test_data_tfidf = tfidf_transformer.transform(test_data_counts)
predicted = clf.predict(test_data_tfidf)
print(predicted)

#use comment model.py

