import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.DataFrame()
df =pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(5)

df.tail()

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

review_lines = list()
lines = df['review'].values.tolist()

for line in lines:
  tokens = word_tokenize(line)
  tokens = [w.lower() for w in tokens]
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in tokens]
  words = [word for word in stripped if word.isalpha()]
  stop_words = set(stopwords.words('english'))
  words = [w for w in words if not w in stop_words]
  review_lines.append(words)

len(review_lines)

import gensim

EMBEDDING_DIM = 100
model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, 
                              window=5, workers=4, min_count=1)
words = list(model.wv.vocab)

filename = 'imdb_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

model.wv.most_similar('sad')

import os

embeddings_index = {}
f = open(os.path.join('', 'imdb_embedding_word2vec.txt'),  encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

X_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

total_reviews = X_train + X_test
max_length = 100

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VALIDATION_SPLIT = 0.2

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)

word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))

review_pad = pad_sequences(sequences, maxlen=max_length)
sentiment =  df['sentiment'].values
print('Shape of review tensor:', review_pad.shape)
print('Shape of sentiment tensor:', sentiment.shape)

indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = review_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]

print('Shape of X_train_pad tensor:', X_train_pad.shape)
print('Shape of y_train tensor:', y_train.shape)

print('Shape of X_test_pad tensor:', X_test_pad.shape)
print('Shape of y_test tensor:', y_test.shape)

EMBEDDING_DIM =100
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(num_words)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.initializers import Constant

model_new = Sequential()
embedding_layer = Embedding(num_words, EMBEDDING_DIM,
                           embeddings_initializer=Constant(embedding_matrix),
                           input_length=max_length, trainable=False)
model_new.add(embedding_layer)
model_new.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model_new.add(MaxPooling1D(pool_size=2))
model_new.add(Flatten())
model_new.add(Dense(1, activation='sigmoid'))

model_new.summary()

model_new.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_new.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)

loss, accuracy = model_new.evaluate(X_test_pad, y_test, batch_size=128)
print("Accuracy: %f" % (accuracy*100))

test_sample_1 = "Hi! How are you today?"
test_sample_2 = "I am fine too. Just busy in doing some work."
test_sample_3 = "Nah. Not really. My job takes a lot of time. It's so boring!"
test_sample_4 = "I always stay at home. I know it is bad for health but who even cares about me!"
test_sample_5 = "I had a dog, it's dead now. I loved it so much!"
test_sample_6 = "Yeah, probably! I like golden retrievers."
test_sample_7 = "You like snakes? That is strange!"
test_sample_8 = "Hahaha. It was nice talking to you. We will talk again. Bye!"
test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

#predict
model_new.predict(x=test_samples_tokens_pad)

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

# define model
model_2 = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)
model_2.add(embedding_layer)
model_2.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
model_2.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model_2.summary())

model_2.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)

score, acc = model_2.evaluate(X_test_pad, y_test, batch_size=128)

print('Test score:', score)
print('Test accuracy:', acc)

print("Accuracy: {0:.2%}".format(acc))

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

#predict
model_2.predict(x=test_samples_tokens_pad)

import pickle

f = open("classifier.pickle", "wb")
pickle.dump(model_2, f)
f.close()

f = open('classifier.pickle', 'rb')
clf = pickle.load(f)
f.close()

clf.predict(x=test_samples_tokens_pad)
