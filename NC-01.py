import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
import pickle

import os
print(os.listdir("/Users/Josh/Documents/Neural_Networks/data_spam_detection"))

import pandas as pd

filepath_dict = {'data':'/Users/Josh/Documents/Neural_Networks/data_spam_detection/DATA.csv'}


df = pd.read_csv("/Users/Josh/Documents//Neural_Networks/data_spam_detection/DATA.csv", sep=',', encoding="ISO-8859-1")


df['label'] = 0
df.loc[df['v1']=='ham', 'label'] = 1
df.loc[df['v1']=='spam','label']= 0


sentences = df['v2'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy (LogReg):", score)

input_dim = X_train.shape[1]  

# format layers

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

# fit the model

model.fit(X_train, y_train, epochs=75, verbose=True, validation_data=(X_test, y_test), batch_size=10)

# save the model to disk

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
