# Gaussian mixture model implemented using tensorflow 2.0

This repository implements a Gaussian mixture model in tensorflow.<br>
Note that the code can quickly be changed to have a GMM as last layer of a neural network<br>
This code can be run with tensorflow GPU to train the GMM accelearted
## Gumbel sampling
Gumbel sampling is used such that the GMM can be used as latent layer for e.g. a variational autoencoder (see ref)

## GMM
Example:

<img src="images/GMM.png" width="600"></img>
## Usage
The GMM.py file can train 2 GMM's. One with label 0 and one with label 1.<br>
To train your GMM put your data in the data folder with numpy array format (NxD) and name: "data_set"<br>
If you have have labels put them in the data folder with name "labels" <br>
If you don't have labels, synthetically make label data by using np.ones((N,))<br>


## Reference
### Gaussian mixture model
http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf
### Gumbel Sampling
https://arxiv.org/pdf/1611.01144.pdf
