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
"""

import logging
from os import remove
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sanitizer import CommentSanitizer


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

    def __init__(self, corpi: list) -> None:
        
        self.load_corpus(corpi)

        self.data = self.standardize_data(self.raw_words)
        self.build_frequency_table(self.data)
        self.remove_oneoffs()

    # add support for multiple corpi
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


    def lemma_and_stem(self, content: list) -> list:
        # TODO: fill this in
        return content


    def standardize_data(self, words: list) -> list:
        """
            load a list of strings pointing to corpus files and concatenate them all, that can be accessed through the 'words' variable 
            
            Args:
            - `words`: list[] - a list of lists in the format [string, meat class (int)]

            Returns:
            - None        
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

            """ now that we're done removing punctuation from our strings, we can make the words into tokens and replace stopwords using that functionality 
                if I try to remove a stopword 'a' from my string, it will just remove all occurances of 'a' from the string, not the stopword version 
                so instead, tokenize the comment and remove it from the list of tokens. 
                ex. 'a big old cat' -> ['a', 'big', 'old', 'cat'] -> ['big', 'old', 'cat'] instead of 'big old ct'
                we also don't need the comments in string form after we're done cleaning them -- they'll be tokenized down the line anyways
            """ 

            comment_tokenized = word_tokenize(comment)

            # now we go through each word of the entry to look for stopwords

            for word in comment_tokenized:
                if word in stopwords.words('english'):
                    # remove it
                    comment_tokenized.remove(word)

            # for now, we're going to skip stemming and lemmatization
            comment_tokenized = self.lemma_and_stem(comment_tokenized)
            new_words.append([comment_tokenized, entry[1]])

        logger.info(f"Ham, Spam: {meat_totals}")

        return new_words

    def remove_oneoffs(self) -> None:

        removal_list = []

        for entry in self.table: 
            if self.table[entry] == [0, 1] or self.table[entry] == [1, 0]:
                removal_list.append(entry)

        # remove them all (can't do in the first loop because of a RuntimeError: dictionary changed size during iteration thingy )
        for entry in removal_list:
            del self.table[entry]

    def build_frequency_table(self, data: list) -> None:
        self.table = {}
        for comment in data:

            # get the class 
            meat_type = comment[1]
            # 
            for word in comment[0]: 
                if word not in self.table:
                    if meat_type == 0:
                        self.table[word] = [1, 0]
                    elif meat_type == 1:
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


     
    def test_x_comments(self, filepath: str, num_comments: int ) -> None:


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
            guessed_class = self.predict(content)

            if guessed_class == correct_class:
                num_correct += 1
        
        logger.info(f"Number correct: {num_correct}/{num_comments}\t\t{round(num_correct / num_comments, 3)}%")




    # TODO: implement TF-IDF
