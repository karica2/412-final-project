#!/usr/bin/env python

from fourtwelve.bow_sklearn import BagOfWordsSKLearn
import time

test_files = ['data/Youtube01-Psy.csv', 'data/Youtube02-KatyPerry.csv', 'data/Youtube03-LMFAO.csv', 'data/Youtube04-Eminem.csv', 'data/Youtube05-Shakira.csv']

print('========= SKLearn BoW ==========')
def print_sklearn_bow(bow, training_time):
    _ = bow.predict(data=bow.comments)
    print(f'{bow}\t\t - {round((bow.get_accuracy()) * 100.0, 2)}% accuracy, {training_time}s to train')

def calc_train_time(bow, smoothing=False):
    start = time.perf_counter()
    bow.train(data=bow.comments, smoothing=smoothing)
    end = time.perf_counter()
    return round(end - start, 2)

def test(test_files):
    bow = BagOfWordsSKLearn(test_files)
    training_time = calc_train_time(bow, False)
    print_sklearn_bow(bow, training_time)
    smoothing_training_time = calc_train_time(bow, True)
    print_sklearn_bow(bow, smoothing_training_time)

print('========= Individual Tests =====')
for test_data in test_files:
    print(f'Test file: {test_data}')
    test(test_files)

print('========= Multiple Datasets ====')
for i in range(len(test_files)):
    print(f'Test files: {test_files[:i+1]}')
    test(test_files[:i+1])

print('========= KFold ================')
bow = BagOfWordsSKLearn(test_files)
bow.train(data=bow.comments)
for i in range(9):
    start = time.perf_counter()
    accuracy = round(bow.kfold(data=bow.comments, k=i+2) * 100, 4)
    training_time = round(time.perf_counter() - start, 4)
    print(f'\tK-Fold k={i+2}, acc={accuracy}, training time={training_time}')
