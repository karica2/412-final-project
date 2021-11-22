from bow_manual import BagOfWords_manual
import logging
from sanitizer import CommentSanitizer

logging.basicConfig(level=logging.INFO)

bow = BagOfWords_manual("../data/Youtube01-Psy.csv")


bow.replace_link_with_constant()
bow.get_frequency_table()

bow.test_x_comments("../data/Youtube01-Psy.csv", 300)
