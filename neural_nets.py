# -*- coding: utf-8 -*-
"""Neural Nets.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PjU5cyvD5wE5qHjVOACOSMxD5KoMhgzg
"""

from bs4 import BeautifulSoup
import re
import pandas as pd
import requests
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import requests
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

Data= pd.read_csv("/content/SentimentalLabels.csv")

Data.head()

MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True,
        stop_words = "english",
        max_features=200
        )
MyDTM = MyCountV.fit_transform(Data["description"])
MyDTM = MyDTM.toarray()

ColumnNames=MyCountV.get_feature_names_out()
MyDTM_DF= pd.DataFrame(MyDTM,columns=ColumnNames)

MyDTM_DF["Labels"] = Data["SentimentalLabels"]

MyDTM_DF

print(MyDTM_DF['Labels'].unique())

data = MyDTM_DF

# Correct the labels
data['Labels'] = data['Labels'].str.strip()  # Remove leading and trailing spaces
data['Labels'] = data['Labels'].replace({'Postive': 'Positive'})  # Correct misspelling

# Check the cleaned unique values to confirm corrections
print(data['Labels'].unique())

Datanew = pd.concat(X)

X = MyDTM_DF.iloc[:, :-1]  # Exclude the label column
y = MyDTM_DF['Labels']

# Convert labels to numerical format
label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
y = y.map(label_mapping)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=3)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(32, activation='relu'),  # Additional hidden layer
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

import numpy as np

# Predict classes on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()