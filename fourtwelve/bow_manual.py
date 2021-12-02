"""
A manual implementation of the Naive Bayes 'bag of words' classifier. 

Authors: 
- Kenan Arica [karica2@uic.edu]
- Rob Schmidt [rschmi2@uic.edu]

"""

import logging
import time
from .sanitizer import CommentSanitizer
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

LINK_CONSTANT = "LINKTOSITE"

class BagOfWords_manual:

    def __init__(self, corpi: list, stemming: bool = False, remove_stopwords: bool = True, remove_oneoffs: bool = True) -> None:

        start = time.time()
        self.load_corpus(corpi)

        self.data = self.standardize_data(self.raw_words, stemming, remove_stopwords)
        self.build_frequency_table(self.data)
        if remove_oneoffs:
            self.remove_oneoffs()

        finished_in = round(time.time() - start, 2)
        logger.info(f"Model trained in {finished_in} seconds")

    def load_corpus(self, corpi: list) -> None:
        """
            load a list of strings pointing to corpus files and concatenate them all, that can be accessed through the 'words' variable 
            
            Args:
            - `corpi`: list[str] - A list of paths to corpus files  

            Returns:
            - None        
        """

        master_table = []

        for corpus in corpi:

            # take our collection of strings, make a parser from each
            CMS = CommentSanitizer(corpus)
            master_table += CMS.parse()
        # we don't actually need most of the information given to us -- just the comment and meat value
        self.raw_words = [[comment["CONTENT"], int(comment["CLASS"])] for comment in master_table]

    def stem(self, content: list) -> list:
        """
            for each word in content, stem it if it can be stemmed 
            
            Args:
            - `content`: list[str] - A list of words  

            Returns:
            - list[str]        
        """

        # we actually don't need to lemmatize the words because NB doesn't care about context 
        porter = PorterStemmer()
        ported = []
        for word in content:
            ported.append(porter.stem(word))

        return ported
        


    def standardize_data(self, words: list, stemming: bool = False, remove_stopwords: bool = True) -> list:
        """
            load a list of strings pointing to corpus files and concatenate them all, that can be accessed through the 'words' variable 
            
            Args:
            - `words`: list[] - a list of lists in the format [string, meat class (int)]
            - `stemming`: bool = False - whether or not to use stemming

            Returns:
            - list of standardized words        
        """
        
        # while we cycle through the data we may as well keep track of how much ham / spam we have 
        meat_totals = [0, 0]
        new_words = []
        for entry in words: 

            comment = entry[0]
            meat_totals[entry[1]] += 1

            # lowercase the data 
            comment = comment.lower()
           
            # for each link, replace it with our constant
            links = re.findall(r'(https?://[^\s]+)', comment)
            comment = re.sub("!|\?|\.|/|,", " ", comment)
            for link in links:
                comment = comment.replace(link, LINK_CONSTANT)

            # break up the content, remove stopwords
            comment_tokenized = word_tokenize(comment)
            if remove_stopwords:

                for word in comment_tokenized:
                    if word in stopwords.words('english'):
                        # remove it
                        comment_tokenized.remove(word)

            if stemming == True:
                comment_tokenized = self.stem(comment_tokenized)
            new_words.append([comment_tokenized, entry[1]])

        self.n_ham = meat_totals[0]
        self.n_spam = meat_totals[1]

        return new_words

    # def sanitize_comment(comment: str) -> list


    def remove_oneoffs(self) -> None:
        """
            remove all words that only occur once in only one class
        
            Returns:
            - None      
        """
        removal_list = []

        for entry in self.table: 
            if self.table[entry] == [0, 1] or self.table[entry] == [1, 0]:
                removal_list.append(entry)

        for entry in removal_list:
            del self.table[entry]

    def build_frequency_table(self, data: list) -> None:
        self.table = {}
        self.total_words_ham = 0
        self.total_words_spam = 0
        for comment in data:

            # get the class 
            meat_type = comment[1]
            # 
            for word in comment[0]: 

                if word not in self.table:
                    if meat_type == 0:
                        self.total_words_ham += 1
                        self.table[word] = [1, 0]
                    elif meat_type == 1:
                        self.total_words_spam += 1
                        self.table[word] = [0, 1]
                else:
                    self.table[word][meat_type] += 1
        
    # pulled these functions from bow_manual like an idiot and modded it so i could see if the original method of prediction was any better using this implementation
    # it's not.

    def predict(self, comment: str) -> int:

            # go through every word and get the probability
            ham = 1
            spam = 1

            comment = self.standardize_data([[comment, -1]])[0]
            
            for token in comment[0]:
                
                # get both frequencies

                # not sure what to do here. If the token isn't in the table, do nothing? need input
                if token not in self.table:
                    continue
                
                entry = self.table[token]
                num_ham = entry[0]
                num_spam = entry[1]
                
                # Get the term frequency
                ham *= (num_ham + 1) / (self.n_ham + 1)
                spam *= (num_spam + 1) / (self.n_spam + 1)
                
            if ham > spam:
                return 0
            if spam > ham:
                return 1

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
