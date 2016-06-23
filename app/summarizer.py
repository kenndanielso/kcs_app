
# coding: utf-8

# ## 0.0 Import modules

# In[1]:

## NLP modules
import gensim
from gensim.models.doc2vec import Doc2Vec
import nltk
import textblob
from textblob import TextBlob
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords ##Note you'll need to download NLTK and corpuses
from spacy.en import English ##Note you'll need to install Spacy and download its dependencies
parser = English()

## Other Python modules
import itertools
from operator import itemgetter
import re
import string
import numpy as np
import pandas as pd
import matplotlib

## Graph module
import networkx as nx

## Machine learning & text vectorizer modules
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ## 1.0 Functions

# ### 1.1 Text pre-processing functions

# In[2]:

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    text = re.sub('[^a-zA-Z ]','',text)
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()
#     text = str(TextBlob(text).correct())
    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):
    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

## Tokenizer specific for Doc2Vec where each word is important, so stop words are not removed.
def doc2vec_tokenizeText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# ### 1.2 Similarity measure functions

# In[3]:

def cos_sim(text1,text2):
    tfvectorizer = TfidfVectorizer(tokenizer=tokenizeText)
    arrays = tfvectorizer.fit_transform([text1,text2]).A
    num = (arrays[0]*arrays[1]).sum()
    denom1 = np.sqrt((arrays[0]**2).sum())
    denom2 = np.sqrt((arrays[1]**2).sum())
    return num/(denom1*denom2)

def similarity(string1,string2):
    w1 = tokenizeText(cleanText(string1))
    w2 = tokenizeText(cleanText(string2))
    score = 0
    for w in w1:
        if w in w2:
            score += 1
        else:
            continue
    for w in w2:
        if w in w1:
            score += 1
        else:
            continue
    return score/(len(w1)+len(w2))

def lDistance(firstString, secondString):
    "Function to find the Levenshtein distance between two words/sentences - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return 1/(distances[-1]+1)


# ### 1.3 Graph building functions

# In[4]:

def buildGraph(nodes,weight):
    "nodes - list of hashables that represents the nodes of the graph"
    gr = nx.Graph() #initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    #add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        if weight == 'cosine':
            edge_weight = cos_sim(firstString, secondString)
        if weight == 'similarity':
            edge_weight = similarity(firstString, secondString)
        if weight == 'ldistance':
            edge_weight = lDistance(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=edge_weight)
    
    return gr


# ### 1.4 Summarizer functions

# In[5]:


def draw_graph(text,weight='cosine'):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())
    graph = buildGraph(sentenceTokens,weight=weight)
    nx.draw_networkx(graph)

def textrank_summarizer(text,weight='cosine',num_sen = 5):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())
    graph = buildGraph(sentenceTokens,weight=weight)

    calculated_page_rank = nx.pagerank(graph, weight='weight')
    
    #Create position scorelist
    #Scores are caluclated such that page rank score is increased by 10% if its the first or last sentence
    #Sentences in the middle of the document are not given increased scores
    pos = np.array(list(range(len(sentenceTokens))))
#     score_array = np.array(1+(abs((pos+0.5) - len(pos)/2)/max((pos+0.5) - len(pos)/2)/10))
#     score_dict = {}
#     for i in range(len(sentenceTokens)):
#         score_dict[sentenceTokens[i]] = score_array[i] 

#     #Adjusts page rank score for position
#     score_adj_page_rank = {k : v * score_dict[k] for k, v in calculated_page_rank.items() if k in score_dict}
    
    #most important sentences in ascending order of importance
#     if pos_score == True:
#         sentences = sorted(score_adj_page_rank, key=score_adj_page_rank.get,reverse=True)
#     else:
#         sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,reverse=True)
  
    #return a word summary
    pos_dict = {}
    for i in range(len(sentenceTokens)):
        pos_dict[sentenceTokens[i]] = pos[i] 

    combined = {k : [v, pos_dict[k]] for k, v in calculated_page_rank.items() if k in pos_dict}

    listlist = []
    for k, v in combined.items():
        listlist.append((k,v[0],v[1]))

    listlist.sort(key=lambda x: x[1],reverse=True)

    summarysentences = listlist[0:num_sen]

    summarysentences.sort(key=lambda x: x[2],reverse=False)

    summary = ""
    for n in range(num_sen):
        summary += ' ' + summarysentences[n][0]
        summary = " ".join(summary.replace(u"\xa0", u" ").strip().split())

    return summary



def doc2vec_summarizer(text,num_sen=5):

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())

    LabeledSentence = gensim.models.doc2vec.LabeledSentence

    sentences = doc2vec_tokenizeText(sentenceTokens)

    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    sentences = labelizeReviews(sentences,'train')

    model = Doc2Vec(min_count=1, window=10, size=500, sample=1e-4, workers=8)
    model.build_vocab(sentences)
    for epoch in range(1000):
        model.train(sentences)


    docvec = []
    for i in range(len(model.docvecs)):
        docvec.append(model.docvecs[i])

    kmeans = KMeans(n_clusters=1)

    kmeans.fit(docvec)

    distance = pairwise_distances(kmeans.cluster_centers_, docvec)

    pos = np.array(list(range(len(sentenceTokens))))

    listlist = [list(x) for x in zip(sentenceTokens,distance.tolist()[0],pos)]

    listlist.sort(key=lambda x: x[1],reverse=False)

    ## Sort by sentence order
    summarysentences = listlist[0:num_sen]

    summarysentences.sort(key=lambda x: x[2],reverse=False)

    summary = ""
    for n in range(num_sen):
        summary += ' ' + summarysentences[n][0]
        summary = " ".join(summary.replace(u"\xa0", u" ").strip().split())

    return summary
        
def lsa_summarizer(text,num_sen=5):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())

    tfvectorizer = TfidfVectorizer(tokenizer=tokenizeText)
    sparse = tfvectorizer.fit_transform(sentenceTokens).A
    lsa = TruncatedSVD(n_components=1)
    concept = lsa.fit_transform(sparse)

    pos = np.array(list(range(len(sentenceTokens))))    
    
    listlist = [list(x) for x in zip(sentenceTokens,concept,pos)]

    listlist.sort(key=lambda x: x[1],reverse=True)

    summarysentences = listlist[0:num_sen]

    summarysentences.sort(key=lambda x: x[2],reverse=False)

    summary = ""
    for n in range(num_sen):
        summary += ' ' + summarysentences[n][0]
        summary = " ".join(summary.replace(u"\xa0", u" ").strip().split())

    return summary

