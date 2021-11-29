"""
A 'bag of words' classifier built upon Sci-kit learn `TfidfVectorizer`

Authors:
- Rob Schmidt <rschmi2@uic.edu>
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import numpy as np

from .sanitizer import CommentSanitizer


class BagOfWordsSKLearn:
    """
    A general BagOfWords implementation that will run on any list of data
    """

    def __init__(self, filenames: 'list[str]'):
        """
        Create a BagOfWords instance which loads multiple CSV files

        Args:
        - `filenames`: list[str] - a list of filenames or paths to CSV
        """

        self._ngram = None
        self._smoothing = None

        comments = []
        for f in filenames:
            cms = CommentSanitizer(f)
            comments.extends(cms.parse())

        comments = sorted(comments, key=lambda c: c['CONTENT'])

        self.comments.extend([x['CONTENT'] for x in comments])
        self.authors.extend([x['AUTHOR'] for x in comments])
        self.meat_quality.extend([int(x['CLASS']) for x in comments])

    def train(self, data: 'list[str]', ngram=1, smoothing=False) -> None:
        """
        Perform BoW on the dataset

        Args:
        - `data`: list[str] - list of strings to BoW
        - `ngram`: int - value to use for ngram (1 for single word, 2 for 2-gram, etc)
        - `smoothing`: bool - flag to use smoothing on TF-IDF Vectorizer
        """

        # tfidf stands for term frequency-inverse document frequency
        # which is a fancy way to say "exclude common stuff and smooth"
        self._vectorizer = TfidfVectorizer(
            use_idf=True,
            smooth_idf=smoothing,
            max_features=10000,
            ngram_range=(ngram,ngram),
            stop_words='english')

        # save the params for string conversion
        self._ngram = ngram
        self._smoothing = smoothing

        # use the vectorizer to fit the data
        self._bag = self._vectorizer.fit_transform(data).toarray()
        self._features = self._vectorizer.get_feature_names_out()

        # split the training and test data from each other
        X_train, self.X_test, Y_train, self.Y_test = train_test_split(self._bag, self.meat_quality)

        # fit the classifier
        self._classifier = GaussianNB()
        self._classifier.fit(X_train, Y_train)

    def transform_data(self, data: 'list[str]') -> 'list[str]':
        """
        Transform data using the BoW TF-IDF vectorizer used in training

        Args:
        - `data`: list[str] - list of strings to transform for usage

        Returns:
        - transformed data according to TF-IDF vectorizer
        """

        return self._vectorizer.transform(data).toarray()

    def get_accuracy(self) -> float:
        """
        Evaluate the GuassianNB model's comment prediction accuracy

        Returns:
        - percentage of correctly predicted comments
        """

        # predict the score of our test data and then compare it with
        # sklearn's accuracy_score() evaluation
        testing = self._classifier.predict(self.X_test)
        return accuracy_score(testing, self.Y_test)

    def get_dataset_ham_spam(self, data: 'list[str]') -> 'tuple[int, int]':
        """
        Get the ham/spam count from the dataset itself

        Args:
        - `data`: list[str] - list of input strings to classify

        Returns:
        - (# ham, # spam) in actual data
        """

        num_spam = np.sum(self.meat_quality)
        num_ham = len(data) - num_spam
        return (num_ham, num_spam)

    def predict(self, data: 'list[str]') -> bool:
        """
        Predict if a collection of input strings are spam

        Args:
        - `data`: list[str] - input string list to predict using GuassianNB model

        Returns:
        - tuple of (# ham comments, # spam comments)
        """

        # transform the data to a usable numpy array
        # and then predict it using the guassian classifier
        usable_data = self.transform_data(data)
        prediction = self._classifier.predict(usable_data)

        # summing the predictions will give # of spam
        # since classes are [0,1]
        spam = np.sum(prediction)
        ham = len(prediction) - spam

        # return the value as a tuple
        return (ham, spam)

    def __str__(self):
        """ Print the class as a string """
        return f"BagOfWords SKLearn - ngram={self._ngram}, smoothing={self._smoothing}"

