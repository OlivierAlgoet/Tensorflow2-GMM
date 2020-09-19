# -*- coding: utf-8 -*-
"""
@author: Olivier Algoet
@summary:
    Causality model defining two gaussian mixture models in tensorflow
    model gives probability of state/observations/other given failure/normal operation
    --> p(x|f) and p(x|n_o)
@legend:
    k cluster amount
    N batch size
    D dimensions of state x | observation z
"""

#Imports
import numpy as np
import tensorflow as tf

#Numerical stability
EPS = 1e-12

class GMM(tf.keras.Model):
    def __init__(self,o_shape,K=10,gumbel_tau=0.1, name="GMM"):
        """
        Parameters
        ----------
        o_shape : Integer
            Output shape
        K : Integer, optional
            Number of Gaussian mixture model clusters The default is 10.
        gumbel_tau : float, optional
            Gumbel temperature The default is 0.001.
            More information: https://arxiv.org/pdf/1611.01144.pdf
        name : Model name, optional
        """
        super().__init__(name=name)
        # Initialalize all the variables
        w_init = tf.random_normal_initializer()
        self.mean_f=tf.Variable(
            initial_value=w_init(shape=(K,o_shape), dtype="float32"),
            trainable=True)
        self.logvar_f=tf.Variable(
            initial_value=w_init(shape=(K,o_shape), dtype="float32"),
            trainable=True)
        self.prob_f=tf.Variable(
            initial_value=w_init(shape=(K,), dtype="float32"),
            trainable=True)
        self.mean_p=tf.Variable(
            initial_value=w_init(shape=(K,o_shape), dtype="float32"),
            trainable=True)
        self.logvar_p=tf.Variable(
            initial_value=w_init(shape=(K,o_shape), dtype="float32"),
            trainable=True)
        self.prob_p=tf.Variable(
            initial_value=w_init(shape=(K,), dtype="float32"),
            trainable=True)
        self.K=K
        self.o_shape=o_shape
        self.tau=gumbel_tau
        self(tf.constant(np.zeros(shape=(1,1), dtype=np.float32)))
        
    def initialize_gmm(self,mean_f,logvar_f,logprob_f,mean_p,logvar_p,logprob_p):
        """
        Initializes the model variables
        E.g. Can firstly use EM Algorithm to avoid local minima
        Parameters
        ----------
        mean_f : float array (KxD)
            means to initialize the GMM for the failure model
        logvar_f : float array (KxD)
            logvar to initialize the GMM for the failure model
        logprob_f : float array (K)
            logprob to initialize the GMM for the failure model
        mean_p : float array (KxD)
            means to initialize the GMM for the normal model
        logvar_p : float array (KxD)
            logvar to initialize the GMM for the normal model
        prob_p : float array (K)
            logvar to initialize the GMM for the normal model
        """
        self.mean_f.assign(mean_f)
        self.logvar_f.assign(logvar_f)
        self.prob_f.assign(logprob_f)
        self.mean_p.assign(mean_p)
        self.logvar_p.assign(logvar_p)
        self.prob_p.assign(logprob_p)
        
    def log_likelihood(self,y,prob,mean,logvar):
        """
        Calculates the log likelihood
        Parameters
        ----------
        y : float array (NxD)
            Data batch from the data_set
        prob : float array (NxK)
            probability of the GMM
        mean : float array (KxNxD)
            mean of the GMM
        logvar : float array (KxNxD)
            logvar of the GMM
        """
        ln2piD = tf.constant(np.log(2 * np.pi) * self.o_shape, dtype=tf.float32)
        sq_distances = tf.math.squared_difference(tf.cast(tf.expand_dims(y, 1),dtype=tf.float32), mean)
        sum_sq_dist_times_inv_var = tf.reduce_sum(sq_distances/(logvar+EPS), 2) 
        log_coefficients =ln2piD + tf.reduce_sum(tf.math.log(logvar+EPS), 2)
        log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_var)
        log_weighted = log_components + tf.math.log(prob)
        exp_log_shifted_sum = tf.reduce_sum(tf.exp(log_weighted),axis=1)
        log_likelihood = tf.reduce_sum(tf.math.log(exp_log_shifted_sum+EPS)) 
        mean_log_likelihood = log_likelihood / tf.cast(tf.shape(y)[0] * tf.shape(y)[1], tf.float32)
        return -mean_log_likelihood
    
    def gumbel_sample(self,prob,mean,logvar):
        """
        Uses gumbel sampling:
            goal = allowing gradient to flow whilst sampling
                   similar to reparametrization trick for gaussians
            see https://arxiv.org/pdf/1611.01144.pdf
        Parameters
        ----------
        prob : float array (NxK)
            probability of GMM (weights)
        mean : float array (KxNxD)
            mean of GMM
        logvar : float array (KxNxD)
            logvar of GMM

        Returns
        -------
        sample : float array
            returns samples equal to the batch size (NxD)

        """
        epsilon=tf.random.uniform(shape=prob.shape, minval=0, maxval=1)
        g=tf.math.log(-tf.math.log(epsilon+EPS)+EPS)
        gumbel_logits=(tf.math.log(prob+EPS)+g)/self.tau
        one_hot=tf.nn.softmax(gumbel_logits)
        gaussian_samples=mean+tf.random.normal(shape=mean.shape) * logvar
        sample=tf.einsum("ij,ijk->ik",one_hot,gaussian_samples)
        return sample
    
    def call(self, inputs):
        """
        Computes Samples and returns the GMM parameters
        using tf.where the decision between failure and normal GMM parameters is made

        Parameters
        ----------
        inputs : float array (Nx1)
            labels for failure or normal operation

        Returns
        -------
        sample: float array (NxD)
            
        prob : float array (NxK)
            
        final_mean : float array (KxNxD)
        
        final_logvar : float array (KxNxD)
            
        """
        batch=inputs.shape[0]
        mean_inputs=tf.ones([batch,self.K,self.o_shape])*tf.expand_dims(inputs,axis=-1)
        prob=tf.where(inputs==1,self.prob_f,self.prob_p)
        mean=tf.where(mean_inputs==1,self.mean_f,self.mean_p)
        logvar=tf.exp(tf.where(mean_inputs==1,self.logvar_f,self.logvar_p))
        prob=tf.nn.softmax(prob)
        return self.gumbel_sample(prob,mean,logvar),prob,mean,logvar

