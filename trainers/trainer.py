import logging
import torch
from torch import nn
from itertools import chain


class Trainer(object):
    """
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, batch_size, num_epochs, learning_rate):

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def train_model(self, model, dataset , output_dir, device , input_dim):
        gd = torch.optim.Adam(
            chain(*[x.parameters() for x in [model.encoder,model.decoder]
                    if (isinstance(x, nn.Module) or isinstance(x, nn.Parameter))]),
            lr=self.learning_rate)
        train_losses = []
        for _ in range(self.num_epochs):
            for i, (batch, _) in enumerate(dataset.train_loader_mnist):
                total = len(dataset.train_loader_mnist)
                gd.zero_grad()
                batch = batch.view(-1, input_dim).to(device)
                loss_value, _ = model.loss(batch)
                loss_value.backward()
                train_losses.append(loss_value.item())
                if (i + 1) % 10 == 0:
                    self._logger.info('Train loss:{}, Batch {} of {} '.format(train_losses[-1],i + 1, total))
                gd.step()
            test_loss = 0.
            for i, (batch, _) in enumerate(dataset.test_loader_mnist):
                batch = batch.view(-1, input_dim).to(device)
                batch_loss, _ = model.loss(batch)
                test_loss += (batch_loss - test_loss) / (i + 1)
            self._logger.info('Test loss after an epoch: {}\n'.format(test_loss))



    def log_and_save_intermediate_outputs(self):
        # may want to save the intermediate results

        pass