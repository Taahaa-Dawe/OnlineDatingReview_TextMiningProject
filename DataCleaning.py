#import praw
from bs4 import BeautifulSoup
import re
import pandas as pd
import requests
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

response = requests.get("https://www.buzzfeed.com/fabianabuontempo/online-dating-success-stories")

soup = BeautifulSoup(response.text,'html.parser')
soup

buzzdata = soup.find_all('span', class_='js-subbuzz__title-text')

buzzdata = buzzdata[1:]

buzzdata = re.sub(r'<span class="js-subbuzz__title-text">',"" ,str(buzzdata))
buzzdata = re.sub(r'</span>',"" ,str(buzzdata))

buzzdata = buzzdata.split('", "')


## news api

url = 'https://newsapi.org/v2/everything'

params = {'q': 'tinder AND bumble AND Online Dating',
    'apiKey': "",
          "language": "en"
               }

request = requests.get(url,params)


newsdata = request.json()

newsdata

title = []
description =[]
source = []

for i in range(len(newsdata["articles"])):
    print(re.sub(r"[^A-Za-z\-]", " ",newsdata["articles"][i]["title"]))

for i in range(len(newsdata["articles"])):
    description.append(re.sub(r"[^A-Za-z\-]", " ",newsdata["articles"][i]["description"]).replace("ul ","").replace("li ","").replace("chars ",""). replace("--",""))
    source.append(re.sub(r"[^A-Za-z\-]", " ",newsdata["articles"][i]["source"]["name"]).replace("com","").replace("org",""))
    title.append(re.sub(r"[^A-Za-z\-]", " ",newsdata["articles"][i]["title"]))

source[1]+","+title[1]+","+description[1]+"\n"

with open("/content/textdata.csv", "a") as new_file :
  new_file.write("label,title,description\n")

with open("/content/textdata.csv", "a") as new_file :
  for i in range(len(description)):
    new_file.write(source[i]+","+title[i]+","+description[i]+"\n")

with open("/content/textdata.csv", "a") as new_file :
  for i in range(len(buzzdata)):
    new_file.write("BuzzFeed"+","+"Success Stories"+","+re.sub(r"[^A-Za-z\-]", " ",buzzdata[i])+"\n")

## reddit
reddit_title = []
reddit_description = []

user_agent = "praw_scraper_1.0"


reddit = praw.Reddit(username="",
                     password="",
                     client_id="",
                     client_secret="",
                     user_agent=user_agent
)
subreddit_name = "OnlineDating"
subreddit = reddit.subreddit(subreddit_name)

for submission in subreddit.new(limit=900):
    reddit_description.append(re.sub(r"[^A-Za-z\-]"," ",submission.selftext))
    reddit_title.append(re.sub(r"[^A-Za-z\-]"," ",submission.title))

with open("/content/textdata.csv", "a") as new_file :
  for i in range(len(reddit_title)):
    new_file.write("Reddit"+","+reddit_title[i]+","+re.sub(r"[^A-Za-z\-]", " ",reddit_description[i])+"\n")

reddit_title2 = []
reddit_description2 = []

subreddit_name = 'AskWomen'
post_id = '10cd6uc'

submission = reddit.submission(id=post_id)

for comment in submission.comments:
    if isinstance(comment, praw.models.Comment):
        reddit_description2.append(comment.body)
        reddit_title2.append(submission.title)

re.sub(r"[^A-Za-z\-]"," ",reddit_description2[0])
re.sub(r"[^A-Za-z\-]"," ",reddit_title2[0])

with open("/content/textdata.csv", "a") as new_file :
  for i in range(len(reddit_title2)):
    new_file.write("Reddit"+","+reddit_title2[i]+","+re.sub(r"[^A-Za-z\-]", " ",reddit_description2[i])+"\n")

data = pd.read_csv("/content/DataText.csv")

data.head()

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

MyCV1 = CountVectorizer(input = "content",
                        stop_words="english")

MyMat = MyCV1.fit_transform(data["description"].str.lower().tolist())

MyCols=MyCV1.get_feature_names_out()

DataFrame = pd.DataFrame(MyMat.toarray(), columns=MyCols)

count = 0
for i in DataFrame.columns:
  if len(i) <= 2:
    count = count +1
    DataFrame.drop(i, axis=1, inplace = True)

print(count)

DataFrame

my_stop_word =["article", "account"]

for i in my_stop_word:
  DataFrame.drop(i, axis=1, inplace = True)

a = " ".join(cat for cat in DataFrame.columns)

DataCopy = DataFrame.copy()

SumofData = DataCopy.sum(axis=0)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

word_cloud = WordCloud(collocations = False, background_color = 'white').generate_from_frequencies(SumofData)

WC1=WordCloud(width=1000, height=600, background_color="white",
               min_word_length=4, #mask=next_image,
               max_words=200).generate_from_frequencies(SumofData)

plt.imshow(WC1, interpolation='bilinear')
plt.axis("off")
plt.show()

LEMMER = WordNetLemmatizer()

STEMMER=PorterStemmer()

def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(word) for word in words]
    return words

def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

MyVect_STEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        lowercase = True,

                        )

Vect_Stem = MyVect_STEM.fit_transform(data["description"].str.lower().tolist())

ColumnNames_s=MyVect_STEM.get_feature_names_out()

CorpusDF_Stem=pd.DataFrame(Vect_Stem.toarray(),columns=ColumnNames_s)
CorpusDF_Stem



count = 0
for i in CorpusDF_Stem.columns:
  if len(i) <= 2:
    count = count +1
    CorpusDF_Stem.drop(i, axis=1, inplace = True)

print(count)

CorpusDF_Stem

DataCopy = CorpusDF_Stem.copy()

DataCopy.drop("----", axis=1)

re.findall(r"[^A-Za-z\-]","")

a = re.findall(r"^[A-Za-z]+","--")
if not a:
  print("true")

print(re.match(r"^[A-Za-z]", "aaron"))

DataCopy.columns

for i in DataCopy.columns:
  Is_pattern =  re.findall(r"^[A-Za-z]+", i)
  if not Is_pattern :
    print(i)
    count = count +1
    DataCopy.drop(i, axis=1, inplace = True)

DataCopy[]

DataCopy.drop("a-okay", axis = 1)

DataCopy

SumofData = DataCopy.sum(axis=0)

a = " ".join(cat for cat in CorpusDF_Stem.columns)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate_from_frequencies(SumofData)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

MyVect_LEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_LEMMER,
                        lowercase = True,

                        )

Vect_LEM = MyVect_STEM.fit_transform(data["description"])

import nltk
nltk.download('wordnet')

MyVect_LEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_LEMMER,
                        lowercase = True,
                        )


Vect_LEM = MyVect_LEM.fit_transform(data["description"].str.lower().tolist())
ColumnNames_lem=MyVect_LEM.get_feature_names_out()
CorpusDF_LEM=pd.DataFrame(Vect_LEM.toarray(),columns=ColumnNames_lem)
CorpusDF_LEM

for i in CorpusDF_LEM.columns:
  if len(i) <= 2:
    CorpusDF_LEM.drop(i, axis = 1, inplace = True)
  elif(re.search(r'[^A-Za-z]+', i)):
    CorpusDF_LEM.drop(i, axis = 1, inplace = True)

CorpusDF_LEM

SumofData = CorpusDF_LEM.sum(axis=0)

a = " ".join(cat for cat in CorpusDF_LEM.columns)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate_from_frequencies(SumofData)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

MyVect_TF=TfidfVectorizer(input='content', stop_words="english")
Vect = MyVect_TF.fit_transform(data["description"])
ColumnNamesTF=MyVect_TF.get_feature_names_out()
CorpusDF_TF=pd.DataFrame(Vect.toarray(),columns=ColumnNamesTF)
CorpusDF_TF

SumofData = CorpusDF_TF.sum(axis=0)

a = " ".join(cat for cat in CorpusDF_TF.columns)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate_from_frequencies(SumofData)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

