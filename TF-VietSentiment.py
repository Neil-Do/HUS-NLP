#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import operator
from vncorenlp import VnCoreNLP
from scipy.sparse import hstack, vstack, csr_matrix, save_npz, load_npz, coo_matrix
import numpy as np

vncorenlp_file = r'Dataset/VnCoreNLP-master/VnCoreNLP-1.1.1.jar'
vncorenlp = VnCoreNLP(vncorenlp_file)
punctutation = "\"#$%&'()*+,-./:;<=>@[\\]^_`{|}~"
print('Created vncorenlp object.')

# data file
positive_input = open("Dataset/SA2016-training_data/SA-training_positive.txt")
negative_input = open("Dataset/SA2016-training_data/SA-training_negative.txt")
neutral_input = open("Dataset/SA2016-training_data/SA-training_neutral.txt")
print("Open data file")

def word_seg_comments(input_file):
    comments = []
    for line in input_file:
        if line == '\n':
            continue
        comment_tokenize = vncorenlp.tokenize(line.strip())
        filtered_punc_comment = []
        for sentence in comment_tokenize:
            sentence = list(filter(lambda word: (word not in punctutation), sentence))
            if len(sentence) > 0:
                filtered_punc_comment.append(sentence)
        comments.append(filtered_punc_comment)
    return comments


positive_comments = word_seg_comments(positive_input)
negative_comments = word_seg_comments(negative_input)
neutral_comments = word_seg_comments(neutral_input)
print('Finished Word Segment')

vncorenlp.close()
print('End VnCoreNLP Object')

def create_bag_of_words(*list_data):
    words_histogram = {}
    for data in list_data:
        for comment in data:
            for sentence in comment:
                for word in sentence:
                    if word in words_histogram:
                        words_histogram[word] += 1
                    else:
                        words_histogram[word] = 1
    sorted_words_histogram = sorted(words_histogram.items(), key=lambda kv: kv[1], reverse=True)
    bag_of_words = {w:i for i,(w,c) in enumerate(sorted_words_histogram)}
    return bag_of_words


bag_of_words = create_bag_of_words(positive_comments, negative_comments, neutral_comments)
print('Created Bag Of Words.')


def tf(comment_vector):
    m = np.amax(comment_vector)
    return np.true_divide(comment_vector, m) * 0.5 + 0.5 # double normalization 0.5 Term Frequency


IDF = np.ones(len(bag_of_words))
COMMENT_NUMBER = 0
print("Created IDF and COMMENT_NUMBER")


def updateIDF(comment_vector):
    global IDF
    for i in range(len(comment_vector)):
        if comment_vector[i] != 0:
            IDF[i] += 1


def vectorize(comment):
    #global COMMENT_NUMBER
    #COMMENT_NUMBER += 1
    comment_vector = np.zeros(len(bag_of_words))
    for sentence in comment:
        for word in sentence:
            comment_vector[bag_of_words[word]] += 1
    #updateIDF(comment_vector)
    return coo_matrix(tf(comment_vector))


def vectorize_all_data(*list_data):
    data_coo = coo_matrix((0, len(bag_of_words)))
    for data in list_data:
        for comment in data:
            comment_to_coo_vector = vectorize(comment)
            data_coo = vstack((data_coo, comment_to_coo_vector))
    return data_coo


# create sparse matrix data
data_coo = vectorize_all_data(positive_comments, negative_comments, neutral_comments)
#IDF = np.log(COMMENT_NUMBER / IDF)
print('Created sparse all data matrix')
print('Computed IDF')
print('Compute TF-IDF')
#data_coo = data_coo.multiply(IDF)
print('Computed TF-IDF')

# create labels array
print('Create labels array')
positive_labels = np.ones(len(positive_comments))
negative_labels = np.ones(len(negative_comments)) * (-1)
neutral_labels = np.zeros(len(neutral_comments))
labels = np.concatenate((positive_labels, negative_labels, neutral_labels))
print('Created labels array')

print('Split to training set, test set')
from sklearn.model_selection import train_test_split
X_train, X_test, label_train, label_test = train_test_split(data_coo, labels, test_size=0.33)

print('Training Model')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1, pre_dispatch=16, cv=5)
grid.fit(X_train, label_train)
print('Finish Train Model')


print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)

lr = grid.best_estimator_
lr.predict(X_test)
print("Score: {:.2f}".format(lr.score(X_test, label_test)))
