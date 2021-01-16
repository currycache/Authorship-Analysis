import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
import math

def remove_unwanted_cols(dataset,cols):
    for col in cols:
        del dataset[col]
    return dataset

def clean_tweets(tweet):
    # wnl = nltk.stem.WordNetLemmatizer()
    # stopwords = nltk.corpus.stopwords.words('english')
    tweet = (unicodedata.normalize('NFKD', tweet).encode('ascii','ignore').decode('utf-8','ignore').lower())
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    words = re.sub(r'[^\w\s]','',tweet).split()
    return words
    # return [wnl.lemmatize(word) for word in words if words not in stopwords]

data = pd.read_csv('../data/jeromepolin2.csv',encoding='iso-8859-9')
jumlahTweetsJ = (len(data.index))

tweetsJ = remove_unwanted_cols(data,['created_at'])

tweetsCleanJ = clean_tweets(''.join(str(tweetsJ['text'].tolist())))

# print(tweetsCleanJ[:300])

data2 = pd.read_csv('../data/ArnoldPoernomo.csv',encoding='iso-8859-9')
jumlahTweetsA = (len(data2.index))
tweetsA = remove_unwanted_cols(data2,['created_at'])

tweetsCleanA = clean_tweets(''.join(str(tweetsA['text'].tolist())))
# print(tweetsCleanA[:300])

tweetsGabung = (tweetsCleanJ + tweetsCleanA)
# print(tweetsGabung[0:150])

numOfWordsJ = dict.fromkeys(tweetsGabung, 0)
for word in tweetsCleanJ:
    numOfWordsJ[word] += 1

numOfWordsA = dict.fromkeys(tweetsGabung, 0)
for word in tweetsCleanA:
    numOfWordsA[word] += 1

# print(numOfWordsA)

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)

    return tfDict

tfJ = computeTF(numOfWordsJ, tweetsCleanJ)
tfA = computeTF(numOfWordsA, tweetsCleanA)
# print (tfJ,tfA)

def computeIDF(documents):

    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

idfs = computeIDF([numOfWordsJ,numOfWordsA])

# print(idfs)

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfJ = computeTFIDF(tfJ, idfs)
tfidfA = computeTFIDF(tfA, idfs)
df = pd.DataFrame([tfidfJ, tfidfA])
# df.to_csv('model1.csv')
# print(df)