#!/usr/bin/env python3

from fourtwelve.bow_manual import BagOfWords_manual
import logging

logging.basicConfig(level=logging.INFO)

bow = BagOfWords_manual("data/Youtube01-Psy.csv")

bow.replace_link_with_constant()
bow.get_frequency_table()

bow.test_x_comments("data/Youtube01-Psy.csv", 300)
