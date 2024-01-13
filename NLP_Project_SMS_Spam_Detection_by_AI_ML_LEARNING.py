#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# In[2]:


import sklearn
print(dir(sklearn.model_selection))


# # 1. Data Gathering

# In[3]:


df = pd.read_csv("SMSSpamCollection.txt", sep = '\t', names = ['Label','Msg'] )
df.head()


# # 2. Exploratory Data analysis

# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df['Label'].value_counts()


# # 3.Data Preprocessing

# In[7]:


corpus = []
lm = WordNetLemmatizer()
for i in range (len(df)):
    review = re.sub('^a-zA-Z0-9',' ',df['Msg'][i])
    review = review.lower()
    review = review.split()
    review = [data for data in review if data not in stopwords.words('english')]
    review = [lm.lemmatize(data) for data in review]
    review = " ".join (review)
    corpus.append(review)
    


# In[8]:


df['Msg'][0]


# In[9]:


stopwords.words('english')


# In[10]:


len(df['Msg'])


# In[11]:


len(corpus)


# In[12]:


df['Msg']= corpus
df.head()


# # 4. Model Building

# # Data spliting

# In[13]:


x = df['Msg']
y = df ['Label']


# In[14]:


x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3, random_state=10)


# In[15]:


len(x_train),len(y_train)


# In[16]:


len(x_test),len(y_test)


# # 4.2 Vectorizer

# In[20]:


tf_obj = TfidfVectorizer()
x_train_tfidf = tf_obj.fit_transform(x_train).toarray()
x_train_tfidf


# In[21]:


x_train_tf_idf.shape


# # 4.3 Pipeline

# In[22]:


text_mnb = Pipeline([('tfidf',TfidfVectorizer()),('mnb',MultinomialNB())])


# In[23]:


text_mnb.fit(x_train, y_train)


# In[28]:


#Accuracy score on testing data
y_test_pred = text_mnb.predict(x_test)
print( 'accuracy_score:', accuracy_score(y_test,y_test_pred))


# In[29]:


#Accuracy score on training data
y_train_pred = text_mnb.predict(x_train)
print( 'accuracy_score:', accuracy_score(y_train,y_train_pred))


# In[32]:


#Classification Report on testing data
y_test_pred = text_mnb.predict(x_test)
print( 'Classification_Report :', classification_report(y_test,y_test_pred))


# In[34]:


#Confusion metrics on testing data
y_test_pred = text_mnb.predict(x_test)
print( 'confusion_matrix :\n', confusion_matrix(y_test,y_test_pred))


# # Prediction on user data

# In[37]:


def preprocess_data(text):
    review = re.sub('^a-zA-Z0-9',' ',text)
    review = review.lower()
    review = review.split()
    review = [data for data in review if data not in stopwords.words('english')]
    review = [lm.lemmatize(data) for data in review]
    review = " ".join (review)
    return[review]
    


# In[41]:


user_data = df['Msg'][2]
print(user_data)
user_data = preprocess_data(user_data)
user_data


# In[48]:


text_mnb.predict(user_data )[0]


# In[53]:


class prediction :
    def __init__ (self,data):
        self.data= data
        
    def user_data_preprocessing(self):
        lm = WordNetLemmatizer()
        review = re.sub('^a-zA-Z0-9',' ',self.data)
        review = review.lower()
        review = review.split()
        review = [data for data in review if data not in stopwords.words('english')]
        review = [lm.lemmatize(data) for data in review]
        review = " ".join (review)
        return[review]   
    def user_data_prediction(self):
        preprocess_data = self.user_data_preprocessing()
        
        if text_mnb.predict(preprocess_data)[0] == 'spam':
            return 'This Message is Spam'
            
        else:
            return 'This Message is Ham'  
    


# In[54]:


df.head()


# In[56]:


user_data = df['Msg'][1]
print(user_data)
prediction(user_data).user_data_prediction()


# In[57]:


user_data = df['Msg'][2]
print(user_data)
prediction(user_data).user_data_prediction()


# In[ ]:




