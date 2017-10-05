# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load data
train_text = pd.read_csv('./data/training_text', sep='\|\|', engine='python', names=['ID','Text'], skiprows=1)
val_text = pd.read_csv('./data/test_text', sep='\|\|', engine='python', names=['ID','Text'], skiprows=1)
train_var = pd.read_csv('./data/training_variants')
val_var = pd.read_csv('./data/test_variants')
val_class = pd.read_csv('./data/stage1_solution_filtered.csv')

# Data check
train_var.info()
train_var.describe()
train_var.head()

val_var.info()
val_var.describe()

train_text.info()
train_text.describe()

val_text.info()
val_text.describe()

X = train_var.iloc[:, :-1]
X['Text'] = train_text['Text']
y = train_var.iloc[:, -1]

val_var = val_var.loc[val_class['ID'], :]
val_var['Text'] = val_text.loc[val_class['ID'], :]

val_class['Class'] = val_class.apply(lambda row: list(np.where(row[1:])[0])[0], axis=1)+1
val_class = val_class.drop(val_class.columns[1:-1], axis=1)

# Calculate Article Length
X['num_words'] = X['Text'].apply(lambda article: len(article.split()))
val_var['num_words'] = val_var['Text'].apply(lambda article: len(article.split()))

X['num_words'].describe()

# Check 1 word long entries
# X[X['num_words'] == 1].head()
X['no_article'] = (X['num_words'] == 1)
val_var['no_article'] = (val_var['num_words'] == 1)

# Split data into training and testing data set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

X_train.Gene.value_counts()
X_train.Variation.value_counts()
y_train.value_counts()
y_train.value_counts(normalize=True)

# Benchmark
from sklearn import metrics

metrics.log_loss(y_train, np.repeat(1/9, 9 * y_train.size).reshape(y_train.size, 9))
# log loss = 2.1972245773362196

# Prior Probabilities
metrics.log_loss(y_test, np.repeat([y_train.value_counts(normalize=True).sort_index()], y_test.size, axis=0).reshape(y_test.size, 9))
# log loss = 1.8170921932092756

### Feature Extraction

# Using Count Vectorizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, preprocessor=None,
                                   stop_words='english', max_features=None)
count_train = count_vectorizer.fit_transform(X_train['Text'])
count_test = count_vectorizer.transform(X_test['Text'])
count_train.shape

count_val = count_vectorizer.transform(val_var['Text'])

# Check out results of bag of words
len(count_vectorizer.get_feature_names())
count_vectorizer.get_feature_names()

# For each, print the vocabulary word and the number of times it
# appears in the training set
count_train_array = count_train.toarray()
count_vocab = count_vectorizer.get_feature_names()
count_dist = np.sum(count_train_array, axis=0)
for tag, count in zip(count_vocab, count_dist):
    if count > 30000:
        print(count, tag)
count_train_array = None
count_vocab = None
count_dist = None

# Using Tf-idf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, preprocessor=None,
#                                    stop_words='english', max_features=None, max_df=0.7)
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train['Text'])
tfidf_test = tfidf_vectorizer.transform(X_test['Text'])

tfidf_val = tfidf_vectorizer.transform(val_var['Text'])

# For each, print the vocabulary word and the number of times it
# appears in the training set
tfidf_train_array = tfidf_train.toarray()
tfidf_vocab = tfidf_vectorizer.get_feature_names()
tfidf_dist = np.sum(tfidf_train_array, axis=0)
for tag, count in zip(tfidf_vocab, tfidf_dist):
    if count > 35:
        print(count, tag)
tfidf_train_array = None
tfidf_vocab = None
tfidf_dist = None

### Train the model

## Using Naive Bayes
# Count Vectorizer Features
from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)
count_pred = nb_classifier.predict(count_test)
count_score = metrics.accuracy_score(y_test, count_pred)
print(count_score)
metrics.confusion_matrix(y_test, count_pred)
metrics.log_loss(y_test, nb_classifier.predict_proba(count_test))

# log loss = 13.857602471924567

# Tf-idf Features
nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_train, y_train)
tfidf_pred = nb_classifier.predict(tfidf_test)
tfidf_score = metrics.accuracy_score(y_test, tfidf_pred)
print(tfidf_score)
metrics.confusion_matrix(y_test, tfidf_pred)
metrics.log_loss(y_test, nb_classifier.predict_proba(tfidf_test))

# Tf-idf Fine-tuning
alphas = np.arange(5, 7.1, 0.5)

# Define train_and_predict()
def nb_train_and_predict(alpha, X_train, X_test, metric = 'accuracy'):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(X_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(X_test)

    if metric == 'accuracy':
        # Compute accuracy: score
        score = metrics.accuracy_score(y_test, pred)
    else:
        # Compute log loss
        score = metrics.log_loss(y_test, nb_classifier.predict_proba(X_test))
    return score

for X_train, X_test, label in [(count_train, count_test, 'Count Vectorizer'), (tfidf_train, tfidf_test, 'Tf-idf Vectorizer')]:
    print(label)
    for alpha in alphas:
        print('Alpha: ', alpha)
        print('Score: ', nb_train_and_predict(alpha, X_train, X_test, metric = 'log_loss'))
        print()

# Tf-idf Features
nb_classifier = MultinomialNB(alpha=6)
nb_classifier.fit(tfidf_train, y_train)
tfidf_pred = nb_classifier.predict(tfidf_test)
tfidf_prob = nb_classifier.predict_proba(tfidf_test)
tfidf_score = metrics.accuracy_score(y_test, tfidf_pred)
print(tfidf_score)
metrics.confusion_matrix(y_test, tfidf_pred)
metrics.log_loss(y_test, tfidf_prob)

# Export prediction to csv
tfidf_df = pd.DataFrame(nb_classifier.predict_proba(tfidf_val), val_var['ID'],
                        columns=['class1','class2','class3','class4','class5','class6','class7','class8','class9'] )
tfidf_df.to_csv("./pred/tfidf_basic.csv", index_label=['ID'])

metrics.log_loss(val_class['Class'], nb_classifier.predict_proba(tfidf_val))
# metrics.accuracy_score(val_class['Class'], nb_classifier.predict(tfidf_val))

# log loss = 1.4197688645866109


# Word2Vec
import nltk.data

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# String cleaning
#from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def article_to_wordlist(article, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    article_text = article
    # 1. Remove HTML
    #article_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    article_text = re.sub("[^a-zA-Z]"," ", article_text)
    # 2.5 Remove single letters
    article_text = re.sub("\s[b-zAHJ-Z]\s", " ", article_text)

    # 3. Convert words to lower case and split them
    words = article_text.lower().split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a article into parsed sentences
def article_to_sentences(article, tokenizer, remove_stopwords=False):
    # Function to split a article into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words

    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(article.strip())

    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(article_to_wordlist( raw_sentence, remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

# Parsing sentences from training set
for article in X["Text"]:
    sentences += article_to_sentences(article, tokenizer)

# Parsing sentences from unlabeled set
for article in val_text["Text"]:
    sentences += article_to_sentences(article, tokenizer)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
# Model training
word2vec_model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
word2vec_model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
word2vec_model_name = "word2vec_genes"
word2vec_model.save(word2vec_model_name)

# word2vec_model.doesnt_match("gene hand chromosome cell".split())
# word2vec_model.doesnt_match("eyes ears nose leg".split())
word2vec_model.most_similar("disease")


## RNN - LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Set up the Keras tokenizer
num_words = 1000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train['Text'].values)

# Pad the training data
X_rnn = tokenizer.texts_to_sequences(X_train['Text'].values)
X_rnn = pad_sequences(X_rnn, maxlen=1000)

# Build out our simple LSTM
embed_dim = 128
lstm_out = 196

# Model saving callback
ckpt_callback = ModelCheckpoint('rnn_lstm',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

rnn_lstm = Sequential()
rnn_lstm.add(Embedding(num_words, embed_dim, input_length = X_rnn.shape[1]))
rnn_lstm.add(LSTM(lstm_out, recurrent_dropout=0.3, dropout=0.3))
rnn_lstm.add(Dense(9, activation='softmax'))
rnn_lstm.compile(optimizer='adam', loss = 'categorical_crossentropy', validation_split=0.2, metrics = ['categorical_crossentropy'])
print(rnn_lstm.summary())

Y_train = pd.get_dummies(y_train).values
# print(X_rnn.shape, Y_train.shape)

batch_size = 32
rnn_lstm.fit(X_rnn, Y_train, epochs=12, batch_size=batch_size, callbacks=[ckpt_callback])

rnn_lstm = load_model('rnn_lstm')

# Pad the data
X_rnn_test = tokenizer.texts_to_sequences(X_test['Text'].values)
X_rnn_test = pad_sequences(X_rnn_test, maxlen=1000)
Y_test = pd.get_dummies(y_test).values

pred_prob = rnn_lstm.predict(X_rnn_test)

pred_indices = np.argmax(pred_prob, axis=1)
classes = np.array(range(1, 10))
preds = classes[pred_indices]
print('Log loss: {}'.format(metrics.log_loss(classes[np.argmax(Y_test, axis=1)], pred_prob)))
print('Accuracy: {}'.format(metrics.accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))

## LinearSVC, TruncatedSVD
