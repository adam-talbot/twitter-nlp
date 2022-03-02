####### NLP PREPARATION SCRIPT #######

import pandas as pd
import unicodedata
import re
import json

import nltk
nltk.download('stopwords')
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


def basic_clean(some_string):
    '''
    Takes in a string and makes all characters lowercased, normalizes unicode characters, and removes any character that is not a letter, number, ', or space
    '''
    some_string = some_string.lower()
    some_string = unicodedata.normalize('NFKD', some_string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    some_string = re.sub(r"[^a-z\s]", '', some_string)
    some_string = re.sub(r"\bhttp\w*", '', some_string)
    return some_string

def tokenize(some_string):
    '''
    Takes in a string and tokenizes it
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    some_string = tokenizer.tokenize(some_string, return_str = True)
    return some_string

def remove_stopwords_set(string, extra_words = ['covid', 'amp', 'pm', 'arizona', 'phoenix', 'tampa', 'florida', 'tx', 'dallas', 'fort', 'worth'], exclude_words = []):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # Create stopword_list.
    stopword_list = stopwords.words('english')
    
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    
    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))

    # Split words in string.
    words = string.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords