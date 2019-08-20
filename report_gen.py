
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import lookup, depression_detect, load_clf
import warnings; warnings.filterwarnings("ignore")
#import time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk #nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import io
import base64

#start = time.time()

#input_tensor = [[m] for m in messages]

#input_tensor = messages

def detect(input_tensor):
    clf, tokenizer_obj = load_clf()
    test_tokens = tokenizer_obj.texts_to_sequences(input_tensor)
    test_tokens_pad = pad_sequences(test_tokens, maxlen=100)

    results = clf.predict(x=test_tokens_pad)
    results = results.tolist()
    results_, _, score_ = depression_detect(results)
    #verdict = lookup(score_)
    return results_, score_

#print(type(input_tensor))
#print(input_tensor)

#results_list = [r[0] for r in results]
#print(results_list)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]"," ", text)
    text = ' '.join(text.split())
    text = text.lower()
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return no_stopword_text

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return " ".join(no_stopword_text)

def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    
    d = words_df.nlargest(columns="count", n=terms)
    
    plt.figure(figsize=(10,10))
    axes = sns.barplot(data=d, x="count", y="word")
    axes.set(ylabel='Word')
    #plt.show()
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    graph_url = base64.b64encode(bytes_image.getvalue()).decode()
    plt.close()
    url = 'data:image/png;base64,{}'.format(graph_url)
    return url
    

def depression_dist(d):
    sns.set(style="whitegrid", palette="muted", color_codes=True)
    #rs = np.random.RandomState(10)
    f, axes = plt.subplots(2, 2, figsize=(7,7), sharex=True)
    sns.despine(left=True)
    #d = rs.normal(size=100)
    print(type(d))
    print(d)
    sns.distplot(d, kde=False, color="b", ax=axes[0,0])
    sns.distplot(d, hist=False, rug=True, color='r', ax=axes[0,1])
    sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
    sns.distplot(d, color="m", ax=axes[1, 1])

    plt.setp(axes, yticks=[])
    plt.tight_layout()
    #plt.show()
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    graph_url = base64.b64encode(bytes_image.getvalue()).decode()
    plt.close()
    url = 'data:image/png;base64,{}'.format(graph_url)
    return url

def depression_trend(x, y):
    sns.set(style="whitegrid")
    f, axes = plt.subplots(figsize=(7,7))
    sns.despine(f, left=True, bottom=True)
    sns.lineplot(x=x, y=y, markers=True, dashes=False)
    sns.scatterplot(x=x, y=y, palette="ch:r=-.2, d=.3_r", ax=axes)
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    graph_url = base64.b64encode(bytes_image.getvalue()).decode()
    plt.close()
    url = 'data:image/png;base64,{}'.format(graph_url)
    return url

'''

#freq_words(messages, 100)
#messages = [(lambda x: remove_stopwords(x)) for x in messages]
#text = " ".join(m for m in messages)
print(text)
text = clean_text(text)
print(text)
text = remove_stopwords(text)
print(text)
#text = ' '.join()
text = [text]
freq_words(text, 100)

end = time.time()

print("Total Time Elapsed:", end-start)

d = np.array([0.45, 0.45, 0.45, 0.38, 0.47, 0.47, 0.45, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.45, 0.38, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.45])
print(type(d))
depression_dist(d)


d = np.array([0.46, 0.1, 0.89, 0.3, 0.99, 0.01])
depression_dist(d)


#d = np.array([0.45, 0.45, 0.45, 0.38, 0.47, 0.47, 0.45, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.45, 0.38, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.45])
d = np.array([0.46, 0.1, 0.89, 0.3, 0.99, 0.01])
#d = np.array([n*100 for n in d])
test = np.arange(len(d))
print(len(d))
print(d)
print(len(test))
print(test)
#y = np.array([0.46, 0.1, 0.89, 0.3, 0.99, 0.7])
#x = np.array([1, 2, 3, 4, 5, 6])
#depression_dist(d)
#line_plot(test, d)
#scatter(test, d)
axes = depression_dist(d)
#axes.plot()
plt.show()



string1 = 'hello'
string2 = 'how are you?'
string3 = 'bye'

inp = input()

if inp.lower() == 'bye':
    print("BYE MF...")

'''