
# coding: utf-8

# In[1]:


# import tools
import pandas as pd
import numpy as np
import tldextract
import warnings

# ignore warning
warnings.simplefilter('ignore')


# In[2]:


# extract the domain
def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return np.nan
    else:
        return ext.domain


# In[3]:


# Alexa Top domain
# https://www.alexa.com/topsites
alexa_df = pd.read_csv('alexa_1M.csv', names=['rank', 'uri'], header=None, encoding='utf-8')

# extract the domain
alexa_df['domain'] = [domain_extract(uri) for uri in alexa_df['uri']]

# delete unnecessary keys
del alexa_df['rank']
del alexa_df['uri']

# drop duplicate and none items
alexa_df = alexa_df.dropna()
alexa_df = alexa_df.drop_duplicates()

# total alexa domains
alexa_total = alexa_df.shape[0]



# In[4]:


# set class
alexa_df['class'] = 'legit'



# In[5]:


# DGA domain
dga_df1 = pd.read_csv('dga_domains.txt',names=['raw_domain'], header=None, encoding='utf-8')
dga_df2 = pd.read_csv('zeus_dga_domains.txt',names=['raw_domain'], header=None, encoding='utf-8')
dga_df3 = pd.read_csv('dga-feed.txt',names=['raw_domain', 'description', 'date', 'source'], header=None, encoding='utf-8')

# delete unnecessary keys
del dga_df3['description']
del dga_df3['date']
del dga_df3['source']

# merge dataframes
dga_df = pd.concat([dga_df1, dga_df2, dga_df3])

# extract the domain and convert to lowercase
dga_df['domain'] = dga_df.applymap(lambda x: x.split('.')[0].strip().lower())

# delete key
del dga_df['raw_domain']

# drop duplicate and none items
dga_df = dga_df.dropna()
dga_df = dga_df.drop_duplicates()

# total dga domains
dga_total = dga_df.shape[0]



# In[6]:


# set class
dga_df['class'] = 'dga'



# In[7]:


# alexa domain
# split training data and test data
training_alexa, testing_alexa = alexa_df[:round(alexa_total*.9)], alexa_df[round(alexa_total*.9):]



# In[8]:


# confirm split is ok
alexa_total == training_alexa.shape[0] + testing_alexa.shape[0]


# In[9]:


# dga domain
# split training data and test data
training_dga, testing_dga = dga_df[:round(dga_total*.9)], dga_df[round(dga_total*.9):]



# In[10]:


# again, confirm split is ok
dga_total == training_dga.shape[0] + testing_dga.shape[0]


# In[11]:


# compact Alexa domain and DGA domain
# ignore the origin index
training_df = pd.concat([training_alexa, training_dga], ignore_index=True)

# shuffle the training data
training_df = training_df.reindex(np.random.permutation(training_df.index))

# total items



# In[12]:


# feature 1: length
training_df['length'] = [len(domain) for domain in training_df['domain']]

# select domains > 4
training_df = training_df[training_df['length']>4]

# after selection



# In[13]:


import math
from collections import Counter

# calculate entropy
# https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

# feature 2: Shannon entropy
training_df['entropy'] = [entropy(domain) for domain in training_df['domain']]
training_df.head()


# In[14]:


# calculate the ratio of single letter
def ratio(lst, s, flag=False):
    total = 0
    for v in lst:
        total += s.count(v)
    if False == flag:
        return total/len(s)
    else:
        return (len(s)-total)/len(s)

# feature 3: vowel
training_df['vowel'] = [ratio('aeiou', domain) for domain in training_df['domain']]

# feature 4: consonant
training_df['consonant'] = [ratio('bcdfghjklmnpqrstvwxyz', domain) for domain in training_df['domain']]

# feature 5: number
training_df['num'] = [ratio('0123456789', domain) for domain in training_df['domain']]

# feature 6: non-alphameric
training_df['non-alphameric'] = [ratio('0123456789abcdefghijklmnopqrstuvwxyz', domain, True) for domain in training_df['domain']]

# show some staff



# In[15]:


# not many domain contains number
training_df[training_df['num']>0].head()


# In[16]:


# word dictionary
word_df = pd.read_csv('words.txt', names=['word'], header=None, dtype={'word': np.str}, encoding='utf-8')

# cleanup the words and convert to lowercase
word_df = word_df[word_df['word'].map(lambda x: str(x).isalpha())]
word_df = word_df.applymap(lambda x: str(x).strip().lower())

# drop duplicate and none items
word_df = word_df.dropna()
word_df = word_df.drop_duplicates()

# show word dictionary



# In[17]:


import sklearn.feature_extraction
import operator

# n-gram
# 3/4/5 char gram
# ignore words that appear between 1e-4 ~  1.0
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
alexa_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)

# combination of fit() and transform() api on same data set
# learn the vocabulary dictionary and return term-document matrix
counts_matrix = alexa_vc.fit_transform(alexa_df['domain'])

# sum up the result array to one-dimensional array
# useful log10
# https://www.zhihu.com/question/22012482
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())

# n-gram list
ngrams_list = alexa_vc.get_feature_names()

# show the result



# In[18]:


# sort and show
# https://docs.python.org/3.7/library/operator.html#operator.itemgetter
sorted_ngrams = sorted(zip(ngrams_list, alexa_counts), key=operator.itemgetter(1), reverse=True)
print('Alexa NGrams: %d' % len(sorted_ngrams))
for ngram, count in sorted_ngrams[:10]:
    print(ngram, count)


# In[19]:


# n-gram on the dictionary words
# same logic
dict_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)

counts_matrix = dict_vc.fit_transform(word_df['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())
ngrams_list = dict_vc.get_feature_names()

# sort iand show
sorted_ngrams = sorted(zip(ngrams_list, dict_counts), key=operator.itemgetter(1), reverse=True)
print('Word NGrams: %d' % len(sorted_ngrams))
for ngram, count in sorted_ngrams[:10]:
    print(ngram, count)


# In[20]:


# document-term matrix
# https://en.wikipedia.org/wiki/Document-term_matrix
# 29106 alexa_grams
alexa_vc.transform(training_df['domain'])


# In[21]:


# transpose index and columns (one-dimensional array)
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.T.html
alexa_vc.transform(training_df['domain']).T


# In[22]:


# 1502402 items
alexa_counts * alexa_vc.transform(training_df['domain']).T


# In[23]:


# feature 7: alexa_grams
training_df['alexa_grams'] = alexa_counts * alexa_vc.transform(training_df['domain']).T

# feature 8: word_grams
training_df['word_grams'] = dict_counts * dict_vc.transform(training_df['domain']).T

# show some staff



# In[24]:


# feature 9: difference between alexa_grams and word_grams
# calculate the difference
training_df['diff'] = training_df['alexa_grams'] - training_df['word_grams']



# In[25]:


# show the feature keys
# first two is: domain, class
training_df.keys()[2:]


# In[26]:


import matplotlib.pyplot as plt

# figure settings
plt.rcParams.update({'figure.max_open_warning': 0, 'axes.grid': True, 'figure.figsize':(14.0, 5.0)})

# plot scatter
def scat(legitX, legitY, dgaX, dgaY, labelX, labelY):
    plt.scatter(legitX, legitY, s=120, c='#aaaaff', label='Alexa', alpha=.1)
    plt.scatter(dgaX, dgaY, s=40, c='r', label='DGA', alpha=.3)
    plt.legend()
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.savefig('plots/'+labelX+labelY)

# split Alexa domain and DGA domain
cond = training_df['class'] == 'dga'
dga = training_df[cond]
legit = training_df[~cond]



# In[63]:


import sklearn.ensemble

# random forests classifier
# 20 trees
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20)

# labels: legit, dga
y = np.array(training_df['class'].tolist())

# use for plot
labels = ['legit', 'dga']




# In[66]:


from itertools import permutations, combinations

# evaluate score by cross-validation
def cross_score(lst):
    X = training_df.as_matrix(lst)
    scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# features: length, entropy, vowel, consonant, num, non-alphameric, alexa_grams, word_grams, diff
# traverse and combine features
keys = training_df.keys()[2:]
n = len(keys)

# result = []
# for i in range(1, n+1):
#     for lst in combinations(keys, i):
#         result.append([cross_score(lst), lst])


# In[67]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# plot confusion matrix
def plot_cm(cm, labels):
    # compute percentanges
    percent = (cm * 100.0) / np.array(np.matrix(cm.sum(axis=1)).T)  # Derp, I'm sure there's a better way

    print('Confusion Matrix Stats')
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))

    # show confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap='coolwarm')
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print('\n\n')


# In[84]:


# test all possible feature combination
def fea_test(domains, lst):
    X = training_df.as_matrix(lst)

    # split training and testing data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit the model
    clf.fit(X_train, y_train)

    # use model to predict
    y_pred = clf.predict(X_test)
    
    # feature keys
    fkeys = training_df.keys()[2:]
    
    # calculate accuracy
    c = 0
    for index, row in domains.iterrows():
        domain, tag = row['domain'], row['class']
        _len = len(domain)
        _entropy = entropy(domain)
        _vowel = ratio('aeiou', domain)
        _con = ratio('bcdfghjklmnpqrstvwxyz', domain)
        _num = ratio('0123456789', domain)
        _non = ratio('', domain, True)
        _alexa_match = alexa_counts * alexa_vc.transform([domain]).T
        _dict_match = dict_counts * dict_vc.transform([domain]).T
        _diff = _alexa_match[0] - _dict_match[0]
        f = dict(zip(fkeys, [_len, _entropy, _vowel, _con, _num, _non, _alexa_match[0], _dict_match[0], _diff]))
        _X = np.array([f.get(key) for key in lst])
        if tag != clf.predict([_X])[0]:
            c += 1
    print(c/len(domains), lst)


# In[75]:


# host, domain, class, subclass
test_df = pd.read_csv('legit-dga_domains.csv', header=0, encoding='utf-8')



# In[76]:


del test_df['host']
del test_df['subclass']

tot = test_df.shape[0]



# In[78]:


# shuffle the test data
test_df = test_df.reindex(np.random.permutation(test_df.index))



# In[80]:



# In[81]:


temp_df = training_df[:12468]


# In[ ]:


# features: length, entropy, vowel, consonant, num, non-alphameric, alexa_grams, word_grams, diff
# traverse and combine features
keys = training_df.keys()[2:]
n = len(keys)

for i in range(1, n+1):
    for lst in combinations(keys, i):
        fea_test(temp_df, lst)


