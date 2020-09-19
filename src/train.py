# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 2020

@author: Olivier Algoet
@summary: This script trains the Gaussian mixture model given example data
"""
# IMPORTS
import numpy as np
import tensorflow as tf
import scipy.stats as stats
import os
from GMM import GMM
from logger import logger

#CONSTANTS
SEED=41
EPOCHS=80
BATCH_SIZE=128
K=6
BAR_LENGTH=40 # only for plot purposes
BETA=1

# set random seed to reproduce
tf.random.set_seed(SEED)
np.random.seed(SEED)

#Get data 
log=logger()
label_name=os.path.join("data","labels")
data_name=os.path.join("data","data_set")
labels=log.unpickle(label_name)
labels=np.expand_dims(labels,axis=-1)
data_set=log.unpickle(data_name)

#Initialize the models and optimizers
gmm_model=GMM(data_set.shape[1],K=K)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) #Low learning rate needed!


#train function
@tf.function #--> optimizes the program by making a graph
def train_models(data,label):
    with tf.GradientTape(persistent=True) as tape: 
        sample,prob,mean,logvar=gmm_model(label)
        log_likelihood=gmm_model.log_likelihood(data,prob,mean,logvar)
    grad = tape.gradient(log_likelihood,gmm_model.variables)
    optimizer.apply_gradients(zip(grad,gmm_model.variables))
    del tape
    return log_likelihood

# Train loop
data_length=len(data_set)
epoch_len=int(len(data_set)/BATCH_SIZE)
for i in range(EPOCHS):
    print("EPOCH {}/{}".format(i,EPOCHS))
    for j in range(epoch_len):
        idxs = np.random.randint(0, data_length, size=BATCH_SIZE)
        label=labels[idxs]
        data=data_set[idxs]
        gmmloss=train_models(tf.constant(data),tf.constant(label))
        bars=int(j*BAR_LENGTH/epoch_len)
        max_len=str(len(str(data_length)))
        print_str="\r{:{}}/{:{}} [{}] -GMM negative likelihood: {:5.4f}".format(j*BATCH_SIZE,max_len,data_length,max_len,"#"*bars+" "*(BAR_LENGTH-bars),gmmloss.numpy())
        end="" if j < epoch_len-1 else "\n"
        print(print_str,end=end)
print("-->Training ended<--")