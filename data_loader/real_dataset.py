import logging
import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms
import torch

class RealDataset(object):
    """

    """
    _logger = logging.getLogger(__name__)

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        
        # Training dataset
        self.train_loader_mnist = torch.utils.data.DataLoader(
            MNIST(root='.', train=True, download=True,
                transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)
        # Test dataset
        self.test_loader_mnist = torch.utils.data.DataLoader(
            MNIST(root='.', train=False, transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)


if __name__ == '__main__':
   

    dataset = RealDataset(100)
    

    
