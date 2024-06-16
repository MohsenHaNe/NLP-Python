#!/usr/bin/env python
# coding: utf-8

# #### This project is about the semantics of text files and deep styling, and it uses many Python packages and gives you a good insight into text processing and NLP.  You can download the used dataset from the kaggle

# In[2]:


import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import os
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model


# In[4]:


stopwords  = stopwords.words('english')


# In[5]:


translator = str.maketrans('','',punctuation)


# In[6]:


pos_doc = []
max_size_pos = 0
for file in os.listdir('./data/pos/'):
    with open('./data/pos/'+file) as f:
        docs = f.read()
        tokened  = word_tokenize(docs)
        punc_clean = [word.translate(translator) for word in tokened]
        stop_cleaned = [word for word in punc_clean if not word in stopwords]
        if max_size_pos < len(stop_cleaned):
            max_size_pos = len(stop_cleaned)
        res = ' '.join(stop_cleaned)
        pos_doc.append(res)
        
        


# In[7]:


neg_doc = []
max_size_neg = 0
for file in os.listdir('./data/neg/'):
    with open('./data/neg/'+file) as f:
        docs = f.read()
        tokened  = word_tokenize(docs)
        punc_clean = [word.translate(translator) for word in tokened]
        stop_cleaned = [word for word in punc_clean if not word in stopwords]
        if max_size_pos < len(stop_cleaned):
            max_size_pos = len(stop_cleaned)
        res_neg = ' '.join(stop_cleaned)
        neg_doc.append(res_neg)
        


# In[8]:


x_train = pos_doc[:800]+neg_doc[:800]
y_train = [1 for i in range(800)] + [0 for i in range(800)] 


# In[9]:


y_train = np.array(y_train)


# In[10]:


x_test = pos_doc[800:]+neg_doc[800:]
y_test = [1 for i in range(200)] + [0 for i in range(200)] 


# In[11]:


#X = pos_doc[:]+neg_doc[:]
#y =  [1 for i in range(1000)] + [0 for i in range(1000)] 


# In[12]:


tokenizer = Tokenizer()


# In[15]:


test_encoded = tokenizer.texts_to_sequences(x_test)
test_padded = pad_sequences(test_encoded,maxlen=max_le,padding='post')


# In[16]:


tokenizer.fit_on_texts(x_train)


# In[14]:


max_le = max(max_size_neg , max_size_pos)
max_le


# In[17]:


vocab_len = len(tokenizer.word_index) + 1
vocab_len


# In[18]:


encoded = tokenizer.texts_to_sequences(x_train)


# In[19]:


padded = pad_sequences(encoded,maxlen=max_le,padding='post')


# In[20]:


padded.shape


# In[ ]:


input_1 = keras.layers.Input(shape=(max_le,))
embedd_1 = keras.layers.Embedding(vocab_len,100)(input_1)
conv_1 = keras.layers.Conv1D(64,10,activation='relu')(embedd_1)
dropout_1 = keras.layers.Dropout(0.5)(conv_1)
maxpool_1 = keras.layers.MaxPool1D(2)(dropout_1)
flatten_1 = keras.layers.Flatten()(maxpool_1)

input_2 = keras.layers.Input(shape=(max_le,))
embedd_2 = keras.layers.Embedding(vocab_len,100)(input_2)
conv_2 = keras.layers.Conv1D(32,4,activation='relu')(embedd_2)
dropout_2 = keras.layers.Dropout(0.3)(conv_2)
maxpool_2 = keras.layers.MaxPool1D(2)(dropout_2)
flatten_2 = keras.layers.Flatten()(maxpool_2)



input_3 = keras.layers.Input(shape=(max_le,))
embedd_3 = keras.layers.Embedding(vocab_len,100)(input_3)
conv_3 = keras.layers.Conv1D(16,8,activation='relu')(embedd_3)
dropout_3 = keras.layers.Dropout(0.2)(conv_3)
maxpool_3 = keras.layers.MaxPool1D(2)(dropout_3)
flatten_3 = keras.layers.Flatten()(maxpool_3)

conc_layer = keras.layers.Concatenate([flatten_1,flatten_2,flatten_3])


dense_1 = keras.layers.Dense(100,activation='relu')(conc_layer)
dense_2 = keras.layers.Dense(50,activation='relu')(dense_1)
out_layer = keras.layers.Dense(1,activation='sigmoid')(dense_2)
model= keras.Model(inputs=[input_1,input_2,input_3],outputs=out_layer)


# In[ ]:


model.summary()


# In[ ]:


plot_model(model)


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


model.fit([padded,padded,padded],np.array(y_train),validation_data=([test_padded,test_padded,test_padded],np.array(y_test)),epochs=10)


# #### Now you can evaluate your model using a test data
