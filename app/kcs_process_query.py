
# coding: utf-8

# ## Import everything

# In[17]:

import pandas as pd
import textblob
from textblob import TextBlob
import dateutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords ##Note you'll need to download NLTK and corpuses
from spacy.en import English ##Note you'll need to install Spacy and download its dependencies
parser = English()
import string
import re
import gensim

import dateutil.parser
import numpy as np
import dateutil
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx

import array
# coding=utf-8

class Status(object):
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad graph type ({})".format(type(graph))
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, {weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc




__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""

__PASS_MAX = -1
__MIN = 0.0000001


def partition_at_level(dendrogram, level):
  
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
 
    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) -                (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph, partition=None, weight='weight', resolution=1.):
 
    dendo = generate_dendrogram(graph, partition, weight, resolution)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph, part_init=None, weight='weight', resolution=1.):
 
    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution)
    new_mod = __modularity(status)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution)
        new_mod = __modularity(status)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):

    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges_iter(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, attr_dict={weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

    return ret


def __load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm
    """
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in graph.nodes():
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in neigh_communities.items():
                incr = resolution * dnc -                        status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
        new_mod = __modularity(status)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key):

    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result


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
    tokens = parser(cleanText(sample))

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


# ## Query

# In[18]:
def query_generator(query_terms=' ',query_date='jan 1, 1950',size=500):


    from elasticsearch import Elasticsearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth
    import pandas as pd 
    from textblob import TextBlob

    if size:
        size = size
    else:
        size = 500

    host = 'search-trial-cc4abhfofwbjogy5nla5sndexq.us-west-2.es.amazonaws.com'
    aws_auth = AWS4Auth('AKIAJQ5JJZM5HDDOMTNQ', 'uVqOuxD+e/iaLkfusHi4TgO1wqrSdUsz2I+VLoAS', 'us-west-2', 'es')

    es = Elasticsearch(
        hosts=[{'host': host, 'port': 443}],
    #     http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )


    # In[19]:
    matches = es.search(index='weiboscope', q=query_terms, size=size)
    hits = matches['hits']['hits']
    query_df = pd.DataFrame(list(pd.DataFrame(hits)['_source']))

    # In[20]:

    ## Convert to date time
    def conv_date(x):
        return dateutil.parser.parse(x)
    
    query_df.date = query_df.date.apply(conv_date)

    try:    
        query_date = dateutil.parser.parse(query_date)
    except:
        query_date = dateutil.parser.parse('jan 1 1950')

    query_df = query_df[query_df['date'] >= query_date].reset_index(drop=True)
    
    if query_df.shape[0] < 1:
        return pd.DataFrame()
    # In[21]:

    ## Add sentiment analysis
    sentiment_score = []
    for row in range(query_df.shape[0]):
        sentiment_score.append(TextBlob(query_df.loc[row,'article']).sentiment.polarity)

    sentiment_score = pd.Series(sentiment_score)

    ##Normalize sentiment score
    sentiment_score = ((sentiment_score + abs(sentiment_score.min()))/(sentiment_score.max()+abs(sentiment_score.min())))

    query_df['sent_score'] = sentiment_score


    # In[22]:

    ## Add length of article
    length = []
    for row in range(query_df.shape[0]):
        length.append(len(query_df.loc[row,'article'].split()))
    query_df['length'] = pd.Series(length)

    ## Limit to articles with over 500 words only
    query_df = query_df[query_df['length']>=500].reset_index(drop=True)


    # In[23]:

    ## Add summary to each article
    summaries = []
    for row in range(query_df.shape[0]):
        summaries.append(gensim.summarization.summarize(query_df.loc[row,'article'],ratio=0.1))
    query_df['summary'] = pd.Series(summaries)


    # In[24]:

    ## Countvectorizer
    tfidfvectorizer = TfidfVectorizer(tokenizer=tokenizeText,strip_accents='unicode',ngram_range=(1,4),min_df=0.01,max_df=0.99,max_features=10000)


    # In[25]:

    tfidf_vector = tfidfvectorizer.fit_transform(query_df.loc[:,'article'])


    # In[26]:

    tfidf_df = pd.DataFrame(tfidf_vector.A,columns=tfidfvectorizer.vocabulary_)


    # In[12]:

    # query_df.columns = ['title_ps','author_ps','date_ps','article_ps','source_ps','sent_score','article_length','summary_ps']


    # In[37]:

    # processed_df = pd.concat((query_df,tfidf_df),axis=1)


    # In[ ]:

    # processed_df.to_pickle('processed_df.pkl')


    # In[3]:

    # processed_df = pd.read_pickle('C:/Users/kennd/Documents/Github/kcs_app/app/processed_df.pkl')


    # In[4]:

    # processed_df1 = processed_df.iloc[:,0:1500]
    # processed_df2 = processed_df.iloc[:,1501:3000]
    # processed_df3 = processed_df.iloc[:,0:3001:4500]
    # processed_df4 = processed_df.iloc[:,0:4501:6000]
    # processed_df5 = processed_df.iloc[:,0:6001:7500]
    # processed_df6 = processed_df.iloc[:,0:7501:9000]
    # processed_df7 = processed_df.iloc[:,0:9001:]

    # from sqlalchemy import create_engine
    # import pandas as pd
    # import json

    # cred = json.load(open('dbcred.json'))

    # engine = create_engine('postgresql://{user}:{password}@{server}/{db}'.format(
    #         user=cred["user"],
    #         password=cred["password"],
    #         server=cred["server"],
    #         db=cred['db']))

    # processed_df1.to_sql('processed_df1',engine,if_exists='replace')
    # processed_df2.to_sql('processed_df2',engine,if_exists='replace')
    # processed_df3.to_sql('processed_df3',engine,if_exists='replace')
    # processed_df4.to_sql('processed_df4',engine,if_exists='replace')
    # processed_df5.to_sql('processed_df5',engine,if_exists='replace')
    # processed_df6.to_sql('processed_df6',engine,if_exists='replace')
    # processed_df7.to_sql('processed_df7',engine,if_exists='replace')


    # ## 2.0 Network Generation

    # ### 2.1 Calculate similarity

    # In[27]:

    ## Network generation
    ## First calculate similarity between articles

    norms = np.sqrt(np.sum(tfidf_vector.A * tfidf_vector.A, axis=1, keepdims=True))  # multiplication between arrays is element-wise

    query_tfidf_normed = tfidf_vector / norms

    weights = np.dot(query_tfidf_normed, query_tfidf_normed.T).tolist()


    # ### 2.2 Generate graph

    # In[28]:

    graph = nx.Graph()
    graph.add_edges_from(
        (i, j, {'weight': weights[i][j]})
        for i in range(tfidf_vector.shape[0]) for j in range(i + 1, tfidf_vector.shape[0]))


    # ### 2.3 Cluster graphs into groupings based on maximum modularity

    # In[29]:

    partition = best_partition(graph,resolution=0.9)

    query_df['group_cluster'] = pd.Series(list(partition.values()))


    # In[35]:

    query_df.columns = ['Text','Author','Date','Source','Title','Sentiment','Length','Summary','Cluster']

    ## Create top words
    new_df = pd.concat((query_df,pd.DataFrame(tfidf_vector.A,columns=tfidfvectorizer.vocabulary_)),axis=1).groupby('Cluster').sum()

    del new_df['Sentiment']
    del new_df['Length']

    clusters = list(query_df['Cluster'].unique())

    cluster = []
    words = []
    for x in clusters:
        cluster.append(x)
        words.append(', '.join(list(new_df.loc[x,:].sort_values(ascending=False)[0:5].index)))

    cluster = pd.Series(cluster)
    words = pd.Series(words)

    cluster_df = pd.concat((cluster,words),axis=1)
    cluster_df.columns = ['Cluster','Cluster Top Words']

    query_df = query_df.merge(cluster_df,on='Cluster',how='left')    
    # In[36]:

    from sqlalchemy import create_engine
    import pandas as pd
    import json

    cred = json.load(open('dbcred.json'))

    engine = create_engine('postgresql://{user}:{password}@{server}/{db}'.format(
            user=cred["user"],
            password=cred["password"],
            server=cred["server"],
            db=cred['db']))

    query_df.to_sql('queried_df',engine,if_exists='replace')

    return query_df[['Title','Author','Date','Summary','Sentiment','Cluster']]
    # In[ ]:



