"""
A manual implementation of the Naive Bayes 'bag of words' classifier. 

Authors: 
- Kenan Arica [karica2@uic.edu]
- Rob Schmidt [rschmi2@uic.edu]

"""

import csv
import logging
from types import prepare_class 
import numpy as np
import matplotlib
from .sanitizer import CommentSanitizer
import re

logger = logging.getLogger(__name__)

LINK_CONSTANT = "LINKTOSITE"

class BagOfWords_manual:

    def __init__(self, filename: str) -> None:
        
        self.CMS = CommentSanitizer(filename)
        self.comments = self.CMS.parse()

    def get_frequency_table(self) -> None:
        
        """
        1. Figure out what type of parameter we need (probably parsed csv, i don't think a numpy array is necessary)
        2. Preprocess the data by getting rid of stop words and replacing links with one master word that indicates a link 
        3. For each word in each comment, update our internal table with a counter as to whether it's from spam or ham

        NOTE: This method allows preprocessing to happen before this function is called, so any stop word removal or link parsing doesn't have to happen here

        internal dict layout: 
            key: [word] value: [tuple(# of occurances in ham, # of occurances in spam)]

        NOTE: class 0: ham, class 1:spam
        """
        table = {}

        # go through 
        for comment in self.comments:

            # split our comment into words
            words = comment["CONTENT"].split()
            spam_class = int(comment["CLASS"])
            # for each word, make or update our entry in the table
            for token in words:

                token = token.lower()
                # if it's not there already, make an entry 
                if token not in table:
                    if spam_class == 0:
                        table[token] = [1, 0]
                    if spam_class == 1:
                        table[token] = [0, 1]

                else:
                    table[token][spam_class] += 1
        # logger.info(table)
        self._table = table

    def replace_link_with_constant(self) -> dict:
        
        comments = self.comments
        for comment in comments:
            
            content = comment["CONTENT"]
            links = re.findall(r'(https?://[^\s]+)', content)
           
            # for each link, replace it with our constant
            for link in links:

                content = content.replace(link, LINK_CONSTANT)

            # strip out any symbols
            # probably a better way of doing this. but i dont care >:)
            content = re.sub("!|\?|\.|/|,", " ", content)

            comment["CONTENT"] = content

        self.comments = comments

        pass
    
    def print_comments(self): 

        for comment in self.comments:
            logger.info(comment)
    

    # TODO: just call this function from replace_link_with_constant

    def clean_test_string(self, input_str: str) -> str:

        links = re.findall(r'(https?://[^\s]+)', input_str)
           
        # for each link, replace it with our constant
        for link in links:

            input_str = input_str.replace(link, LINK_CONSTANT)

        input_string = input_str.lower()
        # strip out any symbols
        # probably a better way of doing this. but i dont care >:)
        content = re.sub("!|\?|\.|/|,", " ", input_string)

        return content

    def predict(self, comment: str) -> int:

        # go through every word and get the probability
        ham = 1
        spam = 1

        comment = self.clean_test_string(comment)
        
        for token in comment.split():
            
            # get both frequencies

            # not sure what to do here. If the token isn't in the table, do nothing? need input
            if token not in self._table:
                continue
            
            entry = self._table[token]
            num_ham = entry[0]
            num_spam = entry[1]
            
            # if the word is a one-off like [0, 1] or [1, 0], ignore it
            # maybe just delete them from the table altogether?
            if entry == [0, 1] or entry == [1, 0]:
                continue

            # not sure what to do about disproportionate words, or words that appear 0 times in one class and lots in the other, because multiplying a probability by 0 will ruin the process.
            # for now, I'm going to weight it as 1 / non-zero + 1 
            if num_ham == 0:
                
                ham *= 1 / (num_spam + 1)
                spam *= 1
                continue
            elif num_spam == 0:
                
                ham *= 1
                spam *= 1 / (num_spam + 1)
                continue

            # if we're all good, calculate
            ham *= num_ham / (num_ham + num_spam)
            spam *= num_spam / (num_spam + num_ham)
            
        if ham > spam:
            return 0
        if spam > ham:
            return 1


        pass

    # could probably move this stuff out into bow_comparison.py
    # test the first X comments from a file
    def test_x_comments(self, filepath: str, num_comments: int ) -> None:


        CMS = CommentSanitizer(filepath)
        comments = CMS.parse()
        if num_comments > len(comments): 
            logger.warn(f" number of comments requested ({num_comments}) is greater than list length ({len(comments)})")
            return
        comments = comments[:num_comments]

        num_correct = 0

        for comment in comments: 

            content = self.clean_test_string(comment["CONTENT"])

            correct_class = int(comment["CLASS"])
            guessed_class = self.predict(content)

            if guessed_class == correct_class:
                num_correct += 1
        
        logger.info(f"Number correct: {num_correct}/{num_comments}\t\t{round(num_correct / num_comments, 3)}%")

