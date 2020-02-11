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

import os
print(os.listdir("/Users/Josh/Documents/data_spam_detection"))

import pandas as pd

filepath_dict = {'data':   '/Users/Josh/Documents/data_spam_detection'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['v1', 'v2'], sep='\t')
    df_list.append(df)

