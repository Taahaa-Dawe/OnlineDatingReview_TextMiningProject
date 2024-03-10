

from sklearn.decomposition import LatentDirichletAllocation
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




Data= pd.read_csv("DataText.csv")




Data.head()





MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True,
        stop_words = "english",
        max_features=200
        )
MyDTM = MyCountV.fit_transform(Data["description"]) 





ColumnNames=MyCountV.get_feature_names_out()








MyDTM1 = MyDTM.toarray()


MyDTM_DF= pd.DataFrame(MyDTM1,columns=ColumnNames)
print(type(MyDTM))
My_Orig_DF=MyDTM_DF



My_Orig_DF.head()



NUM_TOPICS= 2
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=1000, learning_method='online')



lda_Z_DF = lda_model.fit_transform(My_Orig_DF)
print(lda_Z_DF.shape) 

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names_out()[i], topic[i])
                    for i in topic.argsort()[:-top_n - 1:-1]])

print("LDA Model:")
print_topics(lda_model, MyCountV)




print_topics(lda_model, MyCountV)




import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()





panel = pyLDAvis.lda_model.prepare(lda_model, MyDTM, MyCountV)


pyLDAvis.save_html(panel, "ViewsonDating.html")





word_topic = np.array(lda_model.components_)

word_topic = word_topic.transpose()

num_top_words = 15
vocab_array = np.asarray(ColumnNames)


fontsize_base = 20

for t in range(NUM_TOPICS):
    plt.subplot(1, NUM_TOPICS, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.2, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)


plt.savefig("TopicsVis.pdf")








