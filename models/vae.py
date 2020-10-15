import logging
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli, Independent


from helpers.dir_utils import create_dir


class VAE(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, z_dim,num_hidden,input_dim,device):

        self.z_dim = z_dim
        self.num_hidden = num_hidden
        self.input_dim = input_dim    
        self.device = device

        self._build()
        self._logger.debug('Finished building model')

    def _build(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, 2 * self.z_dim)).to(self.device) # note that the final layer outputs real values

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.input_dim)).to(self.device)

    
    def loss(self,x):
        """
        returns
        1. the avergave value of negative ELBO across the minibatch x
        2. and the output of the decoder
        """
        batch_size = x.size(0)
        encoder_output = self.encoder(x)
        pz = Independent(Normal(loc=torch.zeros(batch_size, self.z_dim).to(self.device),
                                scale=torch.ones(batch_size, self.z_dim).to(self.device)),
                        reinterpreted_batch_ndims=1)
        qz_x = Independent(Normal(loc=encoder_output[:, :self.z_dim],
                                scale=torch.exp(encoder_output[:, self.z_dim:])),
                        reinterpreted_batch_ndims=1)
        
        z = qz_x.rsample()
        decoder_output = self.decoder(z)
        px_z = Independent(Bernoulli(logits=decoder_output), 
                        reinterpreted_batch_ndims=1)
        loss = -(px_z.log_prob(x) + pz.log_prob(z) - qz_x.log_prob(z)).mean()
        return loss, decoder_output
            

    @property
    def logger(self):
        try:
            return self._logger
        except:
            raise NotImplementedError('self._logger does not exist!')
    


if __name__ == '__main__':
    pass
