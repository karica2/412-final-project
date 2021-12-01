"""
A bag-of-words implementation using the python3 NLP library, NLTK

IMPORTANT: NLTK needs to download it's stopwords and corpi the first time you run this code. In the interest of not downloading it every time, please use the following lines in a python terminal before running this code:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
Author:
- Kenan Arica [karica2@uic.edu]
- Rob Schmidt <rschmi2@uic.edu>
"""

import collections
import logging
import re
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.text import TextCollection  # can just use tf-idf from here

from .sanitizer import CommentSanitizer


logger = logging.getLogger(__name__)
LINK_CONSTANT = "LINKTOSITE"

"""

The plan:

This implementation is supposed to be an improvement upon the bow_manual.py implementation. The 'manual' implementation does not take advantage
of the powerful text filtering and compuational methods provided by NLTK, so this one will.

* I will keep the link replacement aspect of bow_manual. Replace all links with one common constant, so we can judge the frequency of comments with links and not worry about each link taking dict space.

* Use both TF-IDF and counting, and compare the two in terms of accuracy

"""


class BagOfWords_NLTK:

    def __init__(self, corpi: 'list[str]') -> None:
        """
        Create a BagOfWords implementation using NLTK

        Args:
        - `corpi`: list[str] - list of filenames to load as corpi
        """

        raw_words = self.load_corpus(corpi)
        data = self.standardize_data(raw_words)

        # save the comments data
        # and for convenience save all comments in a list
        self._comments = data
        self._all_comments = [c['CONTENT'] for c in self._comments]

        # TF-IDF stuff doesn't need to be done in the ctor, but
        # this helped debugging at the cost of being a tad bit slower
        self._calculate_all_tf(data)
        self._calculate_idf(data)

    def load_corpus(self, corpi: 'list[str]') -> None:
        """
        Load a list of strings pointing to corpus files and concatenate them all

        Args:
        - `corpi`: list[str] - A list of paths to corpus files

        Returns:
        - list of comments sorted alphabetically by comment message
        """

        master_table = []

        for corpus in corpi:

            # take our collection of strings, make a parser from each
            CMS = CommentSanitizer(corpus)
            master_table += CMS.parse()

        # return the sorted corpi
        return sorted(master_table, key=lambda d: d['CONTENT'])


    def lemma_and_stem(self, content: list) -> list:
        # TODO: fill this in
        return content


    def standardize_data(self, words: 'list[dict]') -> list:
        """
        Strip illegal chars and convert the comments to lowercase

        Args:
        - `words`: list[dict] - a list of dicts as parsed by the CommentSanitizer

        Returns:
        - None
        """

        # while we cycle through the data we may as well keep track of how much ham / spam we have
        meat_totals = [0, 0]
        new_words = []
        for entry in words:

            comment = entry['CONTENT']
            comment_class = int(entry['CLASS'])
            meat_totals[comment_class] += 1

            # lowercase the data
            comment = comment.lower()

            # for each link, replace it with our constant
            links = re.findall(r'(https?://[^\s]+)', comment)
            comment = re.sub(r"[^a-zA-Z0-9 ]", "", comment)
            for link in links:
                comment = comment.replace(link, LINK_CONSTANT)

            # make the content portion of our new_words entry the sanitized comment
            # we just defined above
            entry['CONTENT'] = comment

            """ now that we're done removing punctuation from our strings, we can make the words into tokens and replace stopwords using that functionality
                if I try to remove a stopword 'a' from my string, it will just remove all occurances of 'a' from the string, not the stopword version
                so instead, tokenize the comment and remove it from the list of tokens.
                ex. 'a big old cat' -> ['a', 'big', 'old', 'cat'] -> ['big', 'old', 'cat'] instead of 'big old ct'
                we also don't need the comments in string form after we're done cleaning them -- they'll be tokenized down the line anyways
            """


            # tokenize the comment and get rid of stopwords
            comment_tokenized = word_tokenize(comment)
            # can represent the tokens and stopwords as two sets and take the difference for speed
            comment_tokenized = np.setdiff1d(comment_tokenized, stopwords.words('english'))

            # for now, we're going to skip stemming and lemmatization
            # TODO: Stemm and Lemm
            comment_tokenized = self.lemma_and_stem(comment_tokenized)

            # new_words represents the CommentSanitizer dictionary, but has a new key called
            # 'TOKENS' which represents the tokenized message content
            entry['TOKENS'] = comment_tokenized
            new_words.append(entry)

        # logger.info(f"Ham, Spam: {meat_totals}")
        self.n_ham_comments = meat_totals[0]
        self.n_spam_comments = meat_totals[1]

        # return the new_words list which contains the sanitized comments and their tokenized forms
        return new_words

    # TODO: determine if this is actually necessary, TF-IDF should account for one-offs
#    def remove_oneoffs(self) -> None:
#        removal_list = []
#
#        for entry in self.table:
#            if self.table[entry] == [0, 1] or self.table[entry] == [1, 0]:
#                removal_list.append(entry)
#
#        # remove them all (can't do in the first loop because of a RuntimeError: dictionary changed size during iteration thingy )
#        for entry in removal_list:
#            del self.table[entry]

    def _calculate_tf(self, words: 'list[str]') -> 'dict[str, int]':
        """
        Calculate the term frequencies of all words in a document

        Args:
        - `words`: list[str] - words to count term frequencies of

        Returns:
        - all term frequencies for the document
        """
        freqs = dict.fromkeys(set(words), 0) # freqs for unique words
        occurance_counter = dict(collections.Counter(words))
        for word in words:
            # TF(t,d) = # of times t appears / # of words in d
            # where t is the term, d is the document
            if word in stopwords.words('english'):
                continue
            freqs[word] = int(occurance_counter[word]) / len(words)
        return freqs

    def _calculate_all_tf(self, comments: 'list[dict[str, str]]') -> None:
        """
        Generate a frequency table and occurance table for TF-IDF calculations

        Args:
        - `comments`: list[dict[str, str]] - List of comments as parsed by a sanitizer

        Returns:
        - None
        """

        self._frequency_table = {}

        # place the term frequencies at the index of each document in
        # _frequency_table dictionary
        for comment in comments:
            tf = self._calculate_tf(comment['TOKENS'])
            self._frequency_table[comment['CONTENT']] = tf

    def _calculate_idf(self, comments: 'list[dict[str, str]]') -> None:
        """
        Generate an Inverse Document Frequency table for TF-IDF calculations

        Args:
        - `comments`: list[dict[str, str]] - List of comments as parsed by a sanitizer

        Returns:
        - None
        """

        # combine all comment tokens into one big list
        all_words = []
        for comment in comments:
            all_words.extend(comment['TOKENS'])

        # set some convenient instance variables
        self._idf_table = {}
        self._num_documents = len(comments)
        self._unique_words = set(all_words)

        # calculate IDF for each word
        for word in self._unique_words:
            # creates a list of flags for if a word exists in a document
            # the length of that list is the amount of instances of a word in a document
            instances = len([True for w in self._all_comments if word in w])
            # do IDF math with smoothing
            self._idf_table[word] = np.log((self._num_documents + 1) / (instances + 1)) + 1

    def predict(self, comment: str) -> int:
        """
        Predict a comment's meat quality using a Guassian Naive Bayes model
        """

        # TODO: implement model
        pass

    def _calculate_tf_idf_matrix(self):
        """
        Perform a TF-IDF analysis on the data and classify the result using a Guassian Naive Bayes Model

        Returns:
        - None
        """

        self._tfidf = {}
        for i in range(len(self._comments)):
            for word in self._comments[i]['TOKENS']:
                if word not in self._tfidf:
                    self._tfidf[word] = 0
                self._tfidf[word] += self._frequency_table[self._comments[i]['CONTENT']][word] * self._idf_table[word]

    def _nltk_tf_idf(self) -> None:
        """
        Use NLTK TextCollection to perform TF-IDF
        """
        c = [c['CONTENT'] for c in self._comments]

        t = TextCollection(c)
        tfidf = {}
        for i in range(len(self._comments)):
            for word in self._comments[i]['TOKENS']:
                if word not in tfidf:
                    tfidf[word] = 0
                tfidf[word] += t.tf_idf(word, self._comments[i]['CONTENT'])

        return tfidf

    def test_x_comments(self, filepath: str, num_comments: int, use_tf_idf: bool = False) -> None:
        CMS = CommentSanitizer(filepath)
        comments = CMS.parse()
        if num_comments > len(comments):
            logger.warn(f" number of comments requested ({num_comments}) is greater than list length ({len(comments)})")
            return
        comments = comments[:num_comments]

        num_correct = 0

        for comment in comments:

            content = comment["CONTENT"]
            correct_class = int(comment["CLASS"])
            if use_tf_idf:
                guessed_class = self.tf_idf(content)
            else:
                guessed_class = self.predict(content)

            if guessed_class == correct_class:
                num_correct += 1

        logger.info(f"Number correct: {num_correct}/{num_comments}\t\t{round(num_correct / num_comments, 3)}%")

