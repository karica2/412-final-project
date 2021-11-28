"""Describes the file sanitizer used to cleanup the data from the UCI repo.

It was found that the data had extra characters at the end of the comment message.

Authors:
- Rob Schmidt <rschmi2@uic.edu>
"""

import csv
import logging


logger = logging.getLogger(__name__)


class CommentSanitizer:
    """Sanitize the message of each comment in order to remove extraneous characters"""

    def __init__(self, filename: str):
        """
        Create a sanitizer for the specified CSV file

        Args:
        - `filename`: a string filename which represents the CSV file to load
        """

        self.filename = filename

    def _sanitize_comment(self, line: dict) -> dict:
        """Strip invalid characters from a line in the dataset

        Args:
        - `line`: a single datapoint in the set

        Returns:
        - the same `dict` with sanitized values
        """

        # by using .strip('\ufeff') and .strip() we remove what appears to be an EOL delimiter
        # and any remaining trailing whitespace
        # !The order of the strips is important as trailing whitespace will only be recognized after the EOL is stripped!
        line['CONTENT'] = line['CONTENT'].strip('\ufeff').strip()

        return line

    def parse(self) -> list:
        """Read the CSV file, sanitize the data, and return a list of dictionary items
        indexed by the column header in the CSV

        Returns:
        - a `list` of `dict`s with sanitized data
        """

        dictlist = []

        # read each row from the reader and sanitize it
        # then, append the row to the dictlist
        logger.debug('parsing data from csv at "%s"', self.filename)
        with open(self.filename, errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                r = self._sanitize_comment(row)
                dictlist.append(r)

        logger.debug('parsed %d entries from CSV file "%s"', len(dictlist), self.filename)

        # dictlist should have all rows sanitized now
        return dictlist


