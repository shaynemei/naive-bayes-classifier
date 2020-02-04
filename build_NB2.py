import numpy as np
import os, sys
from collections import Counter

def load_svm(path, vocab, use_pipe=False):
    with open(path) as f:
        num_entries = 0
        for line in f.readlines():
            num_entries += 1
            if not use_pipe:
                line = line.strip().split()
                vocab |= {feature.split(':')[0] for feature in line[1:]}
    vocab = list(vocab)
    vocab.sort()
    
    with open(path) as f:
        data = np.zeros((num_entries, len(vocab) + 1))
        for i, line in enumerate(f.readlines()):
            line = line.strip().split()
            y = y_dict[line[0]]
            data[i][len(vocab)] = y_dict[line[0]]
            features = {feature.split(':')[0]:int(feature.split(':')[1]) for feature in line[1:]}
            for j, word in enumerate(vocab):
                if word in features:
                    data[i][j] = features[word]
    return data, vocab

class NaiveBayes:
    def __init__(self):
        self.len_vocab = None
        self.cls_counts = None
        self.cls_priors = None
        self.cond_probs = None
        
    def fit(self, data, len_vocab, unique_cls_indices, cls_prior_delta, cond_prob_delta):
        self.len_vocab = len_vocab
        self.cls_counts = NaiveBayes.get_cls_counts(data)
        self.cls_priors = NaiveBayes.get_cls_priors(data, self.cls_counts, cls_prior_delta)
        self.cond_probs = NaiveBayes.get_cond_probs(data, self.len_vocab, unique_cls_indices, cond_prob_delta)
        
    
    def predict_inst(self, entry, return_probs):
        prob_0 = np.log10(self.cls_priors[0]) + entry.dot(np.log10(self.cond_probs[0])) 
        prob_1 = np.log10(self.cls_priors[1]) + entry.dot(np.log10(self.cond_probs[1])) 
        prob_2 = np.log10(self.cls_priors[2]) + entry.dot(np.log10(self.cond_probs[2])) 
        
        if return_probs:
            return sorted([(np.float_power(10, prob_0),0), (np.float_power(10, prob_1),1), (np.float_power(10, prob_2),2)], reverse=True)
        
        return np.argmax((prob_0, prob_1, prob_2))
    
    def predict(self, data, return_probs=False):
        res = []
        for entry in data:
            res.append(self.predict_inst(entry, return_probs))
        return res
    
    @staticmethod
    def get_cls_counts(data):
        return Counter(data[:, -1])
    
    @staticmethod
    def get_cls_priors(data, cls_counts, cls_prior_delta):
        cls_num = len(cls_counts)
        return [(cls_counts[i] + cls_prior_delta)/(len(data) + cls_num) for i in range(0, cls_num)] 
    
    @staticmethod
    def get_cond_probs(data, len_vocab, unique_cls_indices, cond_prob_delta):
        # get an array of p(w_i | c_j) of all features in vocab for each class
        # data is sorted by class labels
        data0 = data[:unique_cls_indices[1]]
        data1 = data[unique_cls_indices[1]:unique_cls_indices[2]]
        data2 = data[unique_cls_indices[2]:]
        denom0 = (len_vocab + data0.sum())
        denom1 = (len_vocab + data1.sum())
        denom2 = (len_vocab + data2.sum())
        cond_prob_cls0 = np.array([(cond_prob_delta + np.sum(data0[:, i])) / denom0 for i in range(0, len_vocab)])
        cond_prob_cls1 = np.array([(cond_prob_delta + np.sum(data1[:, i])) / denom1 for i in range(0, len_vocab)])
        cond_prob_cls2 = np.array([(cond_prob_delta + np.sum(data2[:, i])) / denom2 for i in range(0, len_vocab)])
        return (cond_prob_cls0, cond_prob_cls1, cond_prob_cls2)
        
def print_confusion_matrix(res, truth, labels):
    counts_dict = count_res(res, truth)
    print("                  ", end="")
    for label in labels:
        print(f"{label} ", end="")
    print()
    for i, label in enumerate(labels):
        col0 = counts_dict[i] if i in counts_dict else 0
        col1 = counts_dict[i + 3] if i+3 in counts_dict else 0
        col2 = counts_dict[i + 6] if i+6 in counts_dict else 0
        print(f"{label}\t{col0}\t\t{col1}\t\t{col2}")
        
def print_accuracy(res, truth):
    counts_dict = count_res(res, truth)
    col0 = counts_dict[0] if 0 in counts_dict else 0
    col1 = counts_dict[4] if 4 in counts_dict else 0
    col2 = counts_dict[8] if 8 in counts_dict else 0
    print((col0+col1+col2)/len(truth))

def count_res(res, truth):
    unique, counts = np.unique(np.sum([res, truth*3], axis=0), return_counts=True)
    counts_dict = dict()
    for i, j in zip(unique, counts):
        counts_dict[i] = j
    return counts_dict


# usage: build_NB1.sh training_data test_data prior_delta cond_prob_delta model_file sys_output > acc_file
if __name__ == "__main__":
    PATH_TRAIN = sys.argv[1]
    PATH_TEST = sys.argv[2]
    cls_prior_delta = float(sys.argv[3])
    cond_prob_delta = float(sys.argv[4])
    out_model = sys.argv[5]
    out_sys = sys.argv[6]

    y_dict = dict({"talk.politics.guns": 0,
             "talk.politics.mideast": 1,
             "talk.politics.misc": 2})
    labels = list(y_dict.keys())
    vocab = set() 
    train, vocab = load_svm(PATH_TRAIN, vocab)
    test, vocab = load_svm(PATH_TEST, vocab, True)

    train = train[train[:,-1].argsort()]
    unique_cls_indices = np.unique(train[:,-1], return_index=True)[1] 

    nb = NaiveBayes() 
    nb.fit(train, len(vocab), unique_cls_indices, cls_prior_delta, cond_prob_delta) 

    res_train = nb.predict(train[:, :-1])
    res_test = nb.predict(test[:, :-1])
    truth_train = train[:, -1]
    truth_test = test[:, -1]



    with open(out_model, 'w') as f: 
        f.write("%%%%% prior prob P(c) %%%%%\n")
        for i, label in enumerate(labels):
            f.write(f"{label}\t{nb.cls_priors[i]}\t{np.log10(nb.cls_priors[i])}\n")

        f.write("%%%%% conditional prob P(f|c) %%%%%\n")
        for i, label in enumerate(labels):
            f.write(f"%%%%% conditional prob P(f|c) c={label} %%%%%\n")
            for j, word in enumerate(vocab):
                f.write(f"{word}\t{label}\t{nb.cond_probs[i][j]}\t{np.log10(nb.cond_probs[i][j])}\n")

    with open(out_sys, 'w') as f:
        f.write("%%%%% training data:\n")
        for i, probs in enumerate(nb.predict(train[:, :-1], return_probs=True)):
            f.write(f"array:{i} {labels[int(train[i, -1])]} {labels[int(probs[0][1])]} {probs[0][0]} {labels[int(probs[1][1])]} {probs[1][0]} {labels[int(probs[2][1])]} {probs[2][0]}\n")

        f.write("%%%%% test data:\n")
        for i, probs in enumerate(nb.predict(test[:, :-1], return_probs=True)):
            f.write(f"array:{i} {labels[int(test[i, -1])]} {labels[int(probs[0][1])]} {probs[0][0]} {labels[int(probs[1][1])]} {probs[1][0]} {labels[int(probs[2][1])]} {probs[2][0]}\n")

    print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n\n")
    print_confusion_matrix(res_train, truth_train, labels)
    print()
    print("Training accuracy: ", end="")
    print_accuracy(res_train, truth_train)
    print("\n")
    print("Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n")
    print_confusion_matrix(res_test, truth_test, labels)
    print()
    print("Test accuracy: ", end="")
    print_accuracy(res_test, truth_test)






