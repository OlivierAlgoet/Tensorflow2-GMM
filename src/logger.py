# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 2019

@author: Olivier Algoet
"""

try:
   import cPickle as pickle
except:
   import pickle
class logger():
    
    def __init__(self):
        pass
    
    def pickle(self,data,filename):
        """ Pickle data into file specified by filename. """
        pickle.dump(data, open(filename, 'wb'))

    def unpickle(self, filename):
        """ Unpickle data from file specified by filename. """
        try:
            return pickle.load(open(filename, 'rb'))
        except IOError:
            print("error!! not available!")
            return None