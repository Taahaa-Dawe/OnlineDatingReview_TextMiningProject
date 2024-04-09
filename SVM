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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
import numpy as np

!pip install pydotplus
!pip install graphviz
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

from sklearn.svm import LinearSVC
import seaborn as sns

SVM_Linear1 =LinearSVC(C=10)
SVM_Linear1.fit(TrainDF,TrainLabels)
y_pred = SVM_Linear1.predict(TestDF)

cnf_matrix1 = confusion_matrix(TestLabels, y_pred, labels = SVM_Linear1.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=SVM_Linear1.classes_)
disp.plot()

print(classification_report(TestLabels, y_pred))

coef = SVM_Linear1.coef_
negative_coef = coef[0]
neutral_coef = coef[1]
positive_coef = coef[2]

top_negative_index = list(np.argsort(negative_coef,axis=0))
top_negative_words = [BalancedData.columns.tolist()[i] for i in top_negative_index]
top_negative_vals = [negative_coef[i] for i in top_negative_index[:10]]
plt.bar(x=  np.arange(10)  , height=top_negative_vals, color = 'orange')
plt.xticks(np.arange(0, (10)), top_negative_words[:10],rotation=60, ha="right")
plt.gca().invert_yaxis()
plt.title("Negative Features")

top_neutral_index = list(np.argsort(neutral_coef,axis=0))
top_neutral_words = [BalancedData.columns.tolist()[i] for i in top_neutral_index]
top_neutral_vals = [neutral_coef[i] for i in top_neutral_index[:10]]
plt.bar(  x=  np.arange(10)  , height=top_neutral_vals, color = '#B6BCBE')
plt.xticks(np.arange(0, (10)), top_neutral_words[:10],rotation=60, ha="right")
plt.gca().invert_yaxis()
plt.title("Neutral Features")

top_positive_index = list(np.argsort(positive_coef,axis=0))
top_positive_words = [BalancedData.columns.tolist()[i] for i in top_positive_index]
top_positive_vals = [positive_coef[i] for i in top_positive_index[:10]]
plt.bar(  x=  np.arange(10)  , height=top_positive_vals, color = 'BLUE')
plt.xticks(np.arange(0, (10)), top_positive_words[:10], rotation=60, ha="right")
plt.gca().invert_yaxis()
plt.title("Postive Features")

SVM_Linear40 =LinearSVC(C=40)
SVM_Linear40.fit(TrainDF,TrainLabels)
y_pred = SVM_Linear40.predict(TestDF)

cnf_matrix1 = confusion_matrix(TestLabels, y_pred, labels = SVM_Linear1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=SVM_Linear1.classes_)
disp.plot()

print(classification_report(TestLabels, y_pred))

from sklearn.svm import SVC
SVM_rbf = SVC(C = 1,kernel='rbf')
SVM_rbf.fit(TrainDF,TrainLabels)
y_pred = SVM_rbf.predict(TestDF)

cnf_matrix1 = confusion_matrix(TestLabels, y_pred, labels = SVM_Linear1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=SVM_rbf.classes_)
disp.plot()

print(classification_report(TestLabels, y_pred))

from sklearn.svm import SVC
SVM_ploy = SVC(C = 40,kernel='poly',degree = 3)
SVM_ploy.fit(TrainDF,TrainLabels)
y_pred = SVM_ploy.predict(TestDF)

cnf_matrix1 = confusion_matrix(TestLabels, y_pred, labels = SVM_Linear1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=SVM_rbf.classes_)
disp.plot()

print(classification_report(TestLabels, y_pred))

from sklearn.svm import SVC
SVM_ploy = SVC(C = 2,kernel='poly',degree = 3)
SVM_ploy.fit(TrainDF,TrainLabels)
y_pred = SVM_ploy.predict(TestDF)

cnf_matrix1 = confusion_matrix(TestLabels, y_pred, labels = SVM_Linear1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix1,
                               display_labels=SVM_rbf.classes_)
disp.plot()

print(classification_report(TestLabels, y_pred))
