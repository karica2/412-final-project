"""
A bag-of-words implementation using the python3 NLP library, NLTK

Author:
- Kenan Arica [karica2@uic.edu]
"""

import logging


logger = logging.getLogger(__name__)

"""

The plan: 

This implementation is supposed to be an improvement upon the bow_manual.py implementation. The 'manual' implementation does not take advantage 
of the powerful text filtering and compuational methods provided by NLTK, so this one will. 
 
* I will keep the link replacement aspect of bow_manual. Replace all links with one common constant, so we can judge the frequency of comments with links and not worry about each link taking dict space. 

* Use both TF-IDF and counting, and compare the two in terms of accuracy




"""


class BagOfWords_NLTK: 

    def __init__(self) -> None:
        pass
