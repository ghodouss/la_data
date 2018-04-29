import os, sys
import pandas as pd
import numpy as np
import logging, urllib.request, zipfile
from sklearn.preprocessing import LabelEncoder

def random_cv_split(data):
    
    index = np.random.rand(data.shape[0]) > .3
    
    train = data[index]
    cv = data[False == index]
    
    return train, cv


def encode_column(df, column):
    
    label_encoder = LabelEncoder()
    encoded_column = label_encoder.fit_transform(df[column])

    df["Encoded_"+column] = encoded_column
    
    return label_encoder
    

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







