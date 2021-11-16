#!/usr/bin/env python3

"""Main project runner

Description:
- `spam_classifier.py` is used to run the program, and will make extensive
   use of the "fourtwelve" module located within this repo.

Authors:
- Rob Schmidt <rschmi2@uic.edu>
"""

from fourtwelve.sanitizer import CommentSanitizer
import logging
import numpy as np
import sys


# create logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class YTSpamClassifier:
    """Container for all YouTube spam comment classification operations"""

    def _data_to_numpy_array(self, data: list) -> np.ndarray:
        """Convert a list of YouTube comments into a numpy ndarray for use in
        the ML algorithm

        Args:
        - `data`: list of dictionaries of comments indexed by the column names

        Returns:
        - an array of type `np.ndarray`
        """

        pass

    def parse_csv(self, filename: str):
        """Parse a CSV and load the data into this comment analysis object

        Args:
        - `filename`: string filename of the CSV to load
        """

        cs = CommentSanitizer(filename) # use sanitizer to strip extra chars
        self.data = cs.parse()
        self.data_array = self._data_to_numpy_array(self.data)


# main entry point
if __name__ == '__main__':
    if len(sys.argv) != 2:
        # TODO: Load files automatically instead of from CLI
        logger.critical('Invalid usage.\n\tUsage: ./spam_classifier.py [csv filename]')
        exit(127)

    ytsc = YTSpamClassifier()
    ytsc.parse_csv(sys.argv[1])
    print(ytsc.data)

