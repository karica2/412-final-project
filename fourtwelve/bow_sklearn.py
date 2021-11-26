"""
A 'bag of words' classifier built upon Sci-kit learn `TfidfVectorizer`

Authors: 
- Rob Schmidt <rschmi2@uic.edu>
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from .sanitizer import CommentSanitizer


class BagOfWordsSKLearn:
    """
    A general BagOfWords implementation that will run on any list of data
    """

    def __init__(self, filename: str):
        """
        Create a BagOfWords instance which loads a CSV
        
        Args:
        - `filename`: str - filename or path of CSV
        """

        cms = CommentSanitizer(filename)
        parsed_comments = sorted(cms.parse(), key=lambda c: c['CONTENT'])
        self.comments = [x['CONTENT'] for x in parsed_comments]
        self.authors = [x['AUTHOR'] for x in parsed_comments]
        self.meat_quality = [int(x['CLASS']) for x in parsed_comments]

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

        # use the vectorizer to fit the data
        self._bag = self._vectorizer.fit_transform(data).toarray()
        self._features = self._vectorizer.get_feature_names_out()

        # split the training and test data from each other
        X_train, X_test, Y_train, Y_test = train_test_split(self._bag, self.meat_quality)
        self._training_data = (X_train, Y_train)
        self._testing_data = (X_test, Y_test)

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

    def predict(self, string: str) -> bool:
        """
        Predict if an input string is spam

        Args:
        - `string`: str - input string to predict using GuassianNB model
        
        Returns:
        - `1` if input is spam
        - `0` if input is ham
        """

        # transform the data to a usable numpy array
        # and then predict it using the guassian classifier
        data = self.transform_data([string])
        return self._classifier.predict(data)
