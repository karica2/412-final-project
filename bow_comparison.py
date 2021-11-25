#!/usr/bin/env python3

from fourtwelve.bow_manual import BagOfWords_manual
from fourtwelve.sanitizer import CommentSanitizer
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
        
        logger.info(f"Number correct: {num_correct}/{num_comments}\t\t{round(num_correct / num_comments, 3)}%")


bow = BagOfWords_manual("data/Youtube01-Psy.csv")

bow.replace_link_with_constant()
bow.get_frequency_table()

test_x_comments(bow, "data/Youtube01-Psy.csv", 300)
