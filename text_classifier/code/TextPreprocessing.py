from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
#from pycontractions import Contractions
import gensim.downloader as api
import os
from fnmatch import fnmatch
import enchant
from tqdm import tqdm
from multiprocessing import Pool,Lock
import multiprocessing
from nltk.stem import WordNetLemmatizer 
import re

wnl = WordNetLemmatizer() 



nlp = spacy.load('en_core_web_lg')
model = api.load("word2vec-google-news-300")

#cont = Contractions(kv_model=model)
#cont.load_models()


deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

'''
def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = list(cont.expand_texts([text], precise=True))[0]
    return text
'''

def expand_contractions(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    
    if contractions == True: #expand contractions
        try:
            text = expand_contractions(text)
        except:
            text = text
    
    text = text.lower()

    emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",text)
    for email in emails:
        text.replace(email," ")

    ReplaceList = ["|","=",".",",","%","'","+","-","{","}","&","(",")",">","<","*","_","\"","\\","!","@","#","$","^","~","`",":",";","/","?","[","]"]
    for item in ReplaceList:
        text = text.replace(item," ")

    # get word original format
    NewText = ""
    words = text.split()
    for word in words:
        if len(word) > 25 or len(word) <=1:
            continue
        if len(word) == 12 and any(map(str.isdigit, word)):
            continue
        if word.isdigit():
            continue
        if fnmatch(word,"0x*"):
            continue
        newword = wnl.lemmatize(word)
        NewText = NewText +newword+" "
    NewText = NewText+"\n"
    return NewText


def EnglishWordRate(text):
    words = text.split()
    Total = 0.0
    Eng = 0.0
    d = enchant.Dict("en_US")
    for word in words:
        if d.check(word):
            Eng = Eng +1
        Total = Total+1
    return Eng/Total



def PuritySegmens(Segments):

    cleantxt = text_preprocessing(Segments)
    if len(cleantxt.split()) <= 1:
        return ""
    return cleantxt



if __name__ == '__main__':
    
    line = " @brief Initialize the interface with the MCP @param p_hwfn - HW func @param p_ptt - PTT required for register access @return int "
    print(line)
    print("\n")
    newline = PuritySegmens(line)
    print(newline)
