#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pytz import timezone
from datetime import datetime
import numpy as np


from data_loader import RealDataset
from models import VAE
from trainers import Trainer


from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.torch_utils import set_seed, get_device
from helpers.dir_utils import create_dir
from helpers.analyze_utils import sample_vae, plot_samples, plot_reconstructions


def main():
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)

    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    dataset = RealDataset(args.batch_size)
    _logger.info('Finished generating dataset')


    device = get_device()
    model = VAE(args.z_dim,args.num_hidden,args.input_dim,device)


    trainer = Trainer(args.batch_size, args.num_epochs, args.learning_rate)

    
    trainer.train_model(model=model, dataset = dataset, output_dir=output_dir, device = device, input_dim = args.input_dim)

    _logger.info('Finished training model')


    # Visualizations
    samples = sample_vae(model,args.z_dim, device)
    plot_samples(samples)

    plot_reconstructions(model,dataset,device)

    _logger.info('All Finished!')


if __name__ == '__main__':
    main()
