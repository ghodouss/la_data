import os, sys
import pandas as pd
import numpy as np
from gensim.models import word2vec
import gensim
import logging, urllib.request, zipfile


def random_cv_split(data):
    
    index = np.random.rand(data.shape[0]) > .3
    
    train = data[index]
    cv = data[False == index]
    
    return train, cv
    

def string_col_to_int(column, keys, df):
    
    """
    Module to convert a string column
    to a unique integer for quick and 
    dirty correlation mapping
    
    Inputs
    ------------------------
    column: string of dataframe column name
    keys: list of strings containing all vals of column
    df: dataframe to modifiy
    """
    
    for i, key in enumerate(keys):
        
        try:
        
            index = df[column] == key
        except:
            return
        
        df.loc[index, column] = i
        
    return df
        
        
def int_col_to_string(column, keys, df):
    
    """
    Module to convert a int column
    back to a string column
    
    Inputs
    ------------------------
    column: string of dataframe column name
    keys: list of strings containing all vals of column
    df: dataframe to modifiy
    """
        
    for i, key in enumerate(keys):
        
        try:
        
            index = df[column] == i
        except:
            return
        
        df.loc[index, column] = key
    return df







def maybe_download(filename, url):
    """
    Download a file if not present, 
    and make sure it's the right size.
    
    Inputs
    ----------------------------
    filename : string containing filepath
    url : address of data to download
    expected_bytes : int number of bytes download should be
    """
    if not os.path.exists('data/'+filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat('data/'+filename)

    return filename

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def load_gensim_model(root_path, saved_name="mymodel"):
    """
    Module that, given a root path, 
    loads and potentially trains a gensim 
    model on the mattmahoney dataset,
    saves the model as my model
    and returns the trained model
    """
    
    
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url)
    
    if os.path.exists(root_path+saved_name):
        return gensim.models.Word2Vec.load(root_path + "mymodel")
    
    if not os.path.exists((root_path + filename).strip('.zip')):
        zipfile.ZipFile(root_path+filename).extractall()
        
    sentences = word2vec.Text8Corpus((root_path + filename).strip('.zip'))
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    model = word2vec.Word2Vec(sentences, iter=10, min_count=10, size=300, workers=4)

    # save and reload the model
    model.save(root_path + saved_name)
    
    return model 





