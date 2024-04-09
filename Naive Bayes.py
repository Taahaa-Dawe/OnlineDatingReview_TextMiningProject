import pandas as pd
import requests
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import requests
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
import numpy as np

!pip install pydotplus
!pip install graphviz

import nltk
nltk.download('wordnet')

LEMMER = WordNetLemmatizer()
def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(word) for word in words]
    return words

Data = pd.read_csv("/content/SentimentalLabels.csv")

MyVect_LEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_LEMMER,
                        max_features =  250,
                        lowercase = True,


                        )

Vect_LEM = MyVect_LEM.fit_transform(Data["description"].str.lower().tolist())
ColumnNames_lem=MyVect_LEM.get_feature_names_out()
CorpusDF_LEM=pd.DataFrame(Vect_LEM.toarray(),columns=ColumnNames_lem)
CorpusDF_LEM

for i in CorpusDF_LEM.columns:
  if len(i) <= 2:
    CorpusDF_LEM.drop(i, axis = 1, inplace = True)
    print(i)
  elif(re.search(r'[^A-Za-z]+', i)):
    CorpusDF_LEM.drop(i, axis = 1, inplace = True)

Data.head()



CorpusDF_LEM["Labels"] = Data["SentimentalLabels"]

CorpusDF_LEM[CorpusDF_LEM["Labels"]=="Postive"]

CorpusDF_LEM["Labels"] = CorpusDF_LEM["Labels"].str.replace("Postive","Positive")

CorpusDF_LEM[CorpusDF_LEM["Labels"]=="Postive"]

CorpusDF_LEM["Labels"] = CorpusDF_LEM["Labels"].str.replace("Negative ", "Negative")

CorpusDF_LEM["Labels"].value_counts().plot(kind="bar")

CorpusDF_LEM.head()

Negative = CorpusDF_LEM[CorpusDF_LEM["Labels"] =="Negative"].sample(50,random_state = 1)

Positive = CorpusDF_LEM[CorpusDF_LEM["Labels"] =="Positive"]

Neutral = CorpusDF_LEM[CorpusDF_LEM["Labels"] =="Neutral"]

BalancedData  = pd.concat([Negative ,Positive , Neutral])

BalancedData

BalancedData["Labels"].value_counts().plot(kind="bar")

TrainDF, TestDF = train_test_split(BalancedData, test_size=0.30, random_state= 31)

TrainDF.head()

TrainLabels = TrainDF["Labels"]

TestLabels = TestDF["Labels"]

TrainDF.drop("Labels", axis =1, inplace= True)

TestDF.drop("Labels", axis =1, inplace= True)


MyModelNB1= MultinomialNB()
MyModelNB1.fit(TrainDF, TrainLabels)

Prediction1 = MyModelNB1.predict(TestDF)
Prediction1

Proability  = np.round(MyModelNB1.predict_proba(TestDF),2)
Proability

cnf_matrix1 = confusion_matrix(TestLabels, Prediction1)

disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=MyModelNB1.classes_)
disp.plot()

print(classification_report(TestLabels, Prediction1, target_names=MyModelNB1.classes_, zero_division=1))
