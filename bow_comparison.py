#!/usr/bin/env python3

from fourtwelve.bow_nltk import BagOfWords_NLTK
from fourtwelve.bow_manual import BagOfWords_manual
from fourtwelve.bow_sklearn import BagOfWordsSKLearn
from fourtwelve.sanitizer import CommentSanitizer

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_x_comments(bow: BagOfWords_manual, filepath: str, num_comments: int ) -> None:
        """
        test the first X comments from a file
        """

        CMS = CommentSanitizer(filepath)
        comments = CMS.parse()
        if num_comments > len(comments):
            logger.warn(f" number of comments requested ({num_comments}) is greater than list length ({len(comments)})")
            return
        comments = comments[:num_comments]

        num_correct = 0

        for comment in comments:

            content = bow.clean_test_string(comment["CONTENT"])

            correct_class = int(comment["CLASS"])
            guessed_class = bow.predict(content)

            if guessed_class == correct_class:
                num_correct += 1

        print(f"\tNumber correct: {num_correct}/{num_comments}\t\t{round(num_correct / num_comments, 3)}%")

test_file = 'data/Youtube01-Psy.csv'
if len(sys.argv) == 2:
    if sys.argv[1].endswith('.csv'):
        test_file = sys.argv[1]
print(f'Test file: {test_file}\n')

print('========= Manual BoW ==========')
bow = BagOfWords_manual([test_file])

bow.test_x_comments(test_file, 300)


test_files = ['data/Youtube01-Psy.csv', 'data/Youtube02-KatyPerry.csv', 'data/Youtube03-LMFAO.csv', 'data/Youtube04-Eminem.csv', 'data/Youtube05-Shakira.csv']
print(f'\nTraining data:\n\t{test_files}')

print('========= BoW TF-IDF ==========')
bow = BagOfWords_NLTK(test_files)
print('\twithout TF-IDF:')
bow.test_x_comments(test_file, 300, use_tf_idf=False)
#print('\twith TF-IDF:')
#bow.test_x_comments(test_file, 300, use_tf_idf=True)

print('========= SKLearn BoW ==========')
def print_sklearn_bow(bow):
    results = bow.predict(data=bow.comments)
    real = bow.get_dataset_ham_spam(data=bow.comments)

    print(f'{bow}\t\t - {round((bow.get_accuracy()) * 100.0, 2)}% accuracy')
    print(f'\tprediction: {results[0]} ham comments, {results[1]} spam comments')
    print(f'\treal:       {real[0]} ham comments, {real[1]} spam comments')


bow = BagOfWordsSKLearn(test_files)
bow.train(data=bow.comments)
print_sklearn_bow(bow)
bow.train(data=bow.comments, ngram=1, smoothing=True)
print_sklearn_bow(bow)

print('========= SKLearn BoW (k-fold) ==========')
print(f'Average accuracy: {bow.kfold(data=bow.comments)}\n')

print('========= SKLearn BoW (author name) ==========')
bow.train(data=bow.authors, smoothing=True)
print_sklearn_bow(bow)

bow.train(data=bow.authors, ngram=2, smoothing=True)
print_sklearn_bow(bow)
