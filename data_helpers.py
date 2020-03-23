import numpy as np
import re
import itertools
from collections import Counter
from keras.models import Sequential, Model, model_from_json
import pandas as pd
import pickle

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def avg_sentiment(data, row, train_y_cols):
    count = 0
    for col in train_y_cols:
        if data[col][row] == 1:
            count += 1
    is_postive = True if count >= 2 else False
    return is_postive

def load_data_and_labels(train_path, train_x_col, train_y_cols):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples, negative_examples = [], []
    data = pd.read_csv(train_path, sep=",")
    for i in range(len(data[train_y_cols[0]])):
        is_postive = avg_sentiment(data, i, train_y_cols)
        if is_postive is True:
            positive_examples.append(data[train_x_col][i])
        else:
            negative_examples.append(data[train_x_col][i])
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    print("-------------", len(negative_examples), len(positive_examples))
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, sequence_length=1000, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    # total_senteces = sentences + test_sentences
    # sequence_length = max(len(x) for x in total_senteces)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    return padded_sentences


def build_vocab(sentences, out_path):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    save_vocab(vocabulary, vocabulary_inv, out_path)
    return [vocabulary, vocabulary_inv]


def save_vocab(vocabulary, vocabulary_inv, model_path):
    with open(model_path + "/vocabulary", "wb") as f:
        pickle.dump(vocabulary, f)
    with open(model_path + "/vocabulary_inv", "wb") as f:
        pickle.dump(vocabulary_inv, f)


def load_vocab(model_path):
    vocabulary = pickle.load(open(model_path + "/vocabulary", "rb"))
    vocabulary_inv = pickle.load(open(model_path + "/vocabulary_inv", "rb"))
    return [vocabulary, vocabulary_inv]


def save_model(model, model_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path+"/model.json", "w") as json_file:
        json_file.write(model_json)


def load_model(model_path):
    json_file = open(model_path+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model


def build_input_data(sentences, test_sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x, y = build_train_data(sentences, labels, vocabulary)
    test_x= build_test_data(test_sentences, vocabulary)
    return [x, y, test_x]


def build_train_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_test_data(test_sentences, load_path):
    vocabulary, vocabulary_inv = load_vocab(load_path)
    test_x = np.array([[vocabulary[word] if word in vocabulary else vocabulary["<PAD/>"] for word in sentence] for sentence in test_sentences])
    return test_x


def load_train_data(train_path, train_x_col, train_y_cols, save_path):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(train_path, train_x_col, train_y_cols)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded, save_path)
    x, y = build_train_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


def load_test_data(test_path, test_x_col, load_path):
    df = pd.read_csv(test_path, sep='\t', engine='c', encoding='utf-16')
    test_sentences = [clean_str(str(sent)) for sent in df[test_x_col]]
    test_sentences = [s.split(" ") for s in test_sentences]

    padded_test_sentences = pad_sentences(test_sentences)
    test_x = build_test_data(padded_test_sentences, load_path)
    return df, test_x