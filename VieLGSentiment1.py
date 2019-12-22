#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import operator
from vncorenlp import VnCoreNLP
from scipy.sparse import hstack, vstack, csr_matrix, save_npz, load_npz, coo_matrix
import numpy as np
import pandas
import pickle

vncorenlp_file = r'Dataset/VnCoreNLP-master/VnCoreNLP-1.1.1.jar'
vncorenlp = VnCoreNLP(vncorenlp_file)
punctutation = "\"#$%&'()*+,-./:;<=>@[\\]^_`{|}~"

# import Vietnamese Emotion Lexicon Data
df = pandas.read_csv('Dataset/VnEmoLex.csv')
VnEmoLex_dict = {}
d_rows, d_cols = df.shape
count = 0
for index in range(d_rows):
    lex = df['Vietnamese'][index].strip()
    lex = lex.replace(' ', '_')
    VnEmoLex_dict[lex] = 0


# data file
positive_input = open("Dataset/SA2016-training_data/SA-training_positive.txt")
negative_input = open("Dataset/SA2016-training_data/SA-training_negative.txt")
neutral_input = open("Dataset/SA2016-training_data/SA-training_neutral.txt")


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


# positive_comments = word_seg_comments(positive_input)
# negative_comments = word_seg_comments(negative_input)
# neutral_comments = word_seg_comments(neutral_input)

vncorenlp.close()

# positive_segment = open('positive_segment.dat', 'wb')
# negative_segment = open('negative_segment.dat', 'wb')
# neutral_segment = open('neutral_segment.dat', 'wb')
#
# pickle.dump(positive_comments, positive_segment)
# pickle.dump(negative_comments, negative_segment)
# pickle.dump(neutral_comments, neutral_segment)
#
# positive_segment.close()
# negative_segment.close()
# neutral_segment.close()

positive_segment = open('positive_segment.dat', 'r')
negative_segment = open('negative_segment.dat', 'r')
neutral_segment = open('neutral_segment.dat', 'r')

positive_comments = pickle.load(positive_input)
negative_comments = pickle.load(negative_input)
neutral_comments = pickle.load(neutral_input)

positive_segment.close()
negative_segment.close()
neutral_segment.close()

# create bag of vietnamese emotion words in comments dataset
def create_bag_of_words(*list_data):
    for data in list_data:
        for comment in data:
            for sentence in comment:
                for word in sentence:
                    if word in VnEmoLex_dict:
                        VnEmoLex_dict[word] += 1
    sorted_words_histogram = sorted(VnEmoLex_dict.items(), key=lambda kv: kv[1], reverse=True)
    bag_of_words = {w:i for i,(w,c) in enumerate(sorted_words_histogram)}
    return bag_of_words


# bag_of_words = create_bag_of_words(positive_comments, negative_comments, neutral_comments)
#
import json
# json = json.dumps(bag_of_words)
# f = open("bag_of_words.json","w")
# f.write(json)
# f.close()

f = open("bag_of_words.json","r")
bag_of_words = json.loads(f.read())
f.close()

print('Fin save Bag of Words')

def vectorize(comment):
    comment_vector = np.zeros(len(bag_of_words))
    for sentence in comment:
        for word in sentence:
            if word in VnEmoLex_dict:
                comment_vector[bag_of_words[word]] += 1
    return coo_matrix(comment_vector)


def vectorize_all_data(*list_data):
    data_coo = coo_matrix((0, len(bag_of_words)))
    for data in list_data:
        for comment in data:
            comment_to_coo_vector = vectorize(comment)
            data_coo = vstack((data_coo, comment_to_coo_vector))
    return data_coo


# create sparse matrix data
data_coo = vectorize_all_data(positive_comments, negative_comments, neutral_comments)

# create labels array
positive_labels = np.ones(len(positive_comments))
negative_labels = np.ones(len(negative_comments)) * (-1)
neutral_labels = np.zeros(len(neutral_comments))
labels = np.concatenate((positive_labels, negative_labels, neutral_labels))

# split dataset to train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, label_train, label_test = train_test_split(data_coo, labels, test_size=0.33)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1, pre_dispatch=16, cv=5)
grid.fit(X_train, label_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)

lr = grid.best_estimator_
lr.predict(X_test)
print("Score: {:.2f}".format(lr.score(X_test, label_test)))
