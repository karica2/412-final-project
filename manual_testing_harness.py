from numpy import number
from fourtwelve.bow_manual import BagOfWords_manual
from fourtwelve.sanitizer import CommentSanitizer

import time
import sys
import logging as logger

logger.basicConfig(level=logger.INFO)
# logger = logging.getLogger(__name__)

def test_x_comments(bow: BagOfWords_manual, filepath: str, num_comments: int) -> int:
        """
        test the first X comments from a file
        """
        start = time.time()

        CMS = CommentSanitizer(filepath)
        comments = CMS.parse()
        if num_comments > len(comments):
            logger.warn(f" number of comments requested ({num_comments}) is greater than list length ({len(comments)})")
            return
        comments = comments[:num_comments]

        num_correct = 0

        for comment in comments:
            
            # content = bow.standardize_data(comment["CONTENT"])
            content = comment["CONTENT"]
            correct_class = int(comment["CLASS"])
            guessed_class = bow.predict(content)

            if guessed_class == correct_class:
                num_correct += 1
        finished_in = round(time.time() - start, 2)
        print(f"\tNumber correct: {num_correct}/{num_comments}\t\t{round(num_correct / num_comments, 3)}%")
        print(f"Predicted {num_comments} comments in {finished_in} seconds")
        return num_correct


test_files = ['data/Youtube01-Psy.csv', 'data/Youtube02-KatyPerry.csv', 'data/Youtube03-LMFAO.csv', 'data/Youtube04-Eminem.csv', 'data/Youtube05-Shakira.csv']


sample_size = 300

def unit_test(test_number: int, use_stemming: bool, remove_stopwords: bool, remove_oneoffs: bool, include_test_set: bool):
    """
    Run a unit test based on the given parameters
    """
    number_correct = 0
    print(f"======== Test {test_number}: stemming = {use_stemming}, remove_stopwords = {remove_stopwords}, remove_oneoffs = {remove_oneoffs}, include_test_set = {include_test_set} ========")
    start = time.time()
    # testing without stemming, with no stopwords, on each 4
    for x in range(len(test_files)): 

        dataset = test_files[x]
        training_set = [y for y in test_files]
        if not include_test_set:
            training_set.remove(dataset)

        bow = BagOfWords_manual(training_set, stemming=use_stemming, remove_stopwords=remove_stopwords, remove_oneoffs=remove_oneoffs)

        logger.debug(f"Testing set: {dataset}")
        logger.debug(f"Training set: {training_set}")

        logger.debug(f"\n========= Manual Bag Of Words ===========")
        number_correct += test_x_comments(bow, dataset, sample_size)

    runtime = round(time.time() - start, 2)
    print(f"Average accuracy across test {test_number}: {round((number_correct / 1500) * 100, 2)}%")
    print(f"Runtime for test {test_number}: {runtime} seconds")

def permutation(number_of_bools: int):
    """ 
    get all permutations of x booleans 
    """
    big_range = 2 ** number_of_bools

    configs = [[] for x in range(big_range)]
    val = True
    for y in range(number_of_bools):
        # flip after y 
        flip_after = int(big_range / (2 ** (y + 1)))
        inc = 0
        for x in range(big_range):
            configs[x].append(val)
            inc += 1
            if inc >= flip_after:
                val = not val
                inc = 0
    return configs

configs = permutation(3)


for x in range(len(configs)):
    
    current_config = configs[x]
    unit_test(x, current_config[0], current_config[1], current_config[2], False)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# include testing data in training set. will always have better results obvi

for x in range(len(configs)):
    
    current_config = configs[x]
    unit_test(x, current_config[0], current_config[1], current_config[2], True)



