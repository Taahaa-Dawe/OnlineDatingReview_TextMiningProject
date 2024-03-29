

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
import nltk
nltk.download('wordnet')

data = pd.read_csv("/content/DataText.csv")

LEMMER = WordNetLemmatizer()

def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(word) for word in words]
    return words

MyVect_LEM=TfidfVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_LEMMER,
                        lowercase = True,
                        max_features= 600
                        )

Vect_LEM = MyVect_LEM.fit_transform(data["description"].str.lower().tolist())
ColumnNames_lem=MyVect_LEM.get_feature_names_out()
CorpusDF_LEM=pd.DataFrame(Vect_LEM.toarray(),columns=ColumnNames_lem)
CorpusDF_LEM

CorpusDF_LEM.columns

my_stop_word = ['removed']

for i in my_stop_word:
  CorpusDF_LEM.drop(i, inplace=True, axis= 1)

for i in CorpusDF_LEM.columns:
    if i == "bf":
        CorpusDF_LEM.rename({"bf":"boyfriend"})
        print("Changed")
    elif len(i) <3:
        CorpusDF_LEM.drop(i, inplace=True, axis= 1)

CorpusDF_LEM

transaction_data = []

# Iterate through each row in the DataFrame
for index, row in CorpusDF_LEM.iterrows():
    # Initialize an empty set to store items for the current transaction
    transaction = set()
    # Iterate through each column in the row
    for column in CorpusDF_LEM.columns:
        # If the value in the current cell is non-zero, add the column name to the transaction
        if row[column] != 0:
            transaction.add(column)

    transaction_data.append(transaction)

# Print the transactional data
for i, transaction in enumerate(transaction_data):
    print(f"Transaction {i + 1}: {transaction}")

for i, transaction in enumerate(transaction_data):
    if transaction == set():
      transaction_data.remove(set())

for i, transaction in enumerate(transaction_data):
    print(f"Transaction {i + 1}: {transaction}")

transaction_df = pd.DataFrame(transaction_data)

transaction_df.to_csv("TransactionData.csv")

transaction_df

