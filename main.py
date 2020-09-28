#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pytz import timezone
from datetime import datetime
import numpy as np

from data_loader import SyntheticDataset
from models import GAE
from trainers import ALTrainer
from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
# from helpers.torch_utils import set_seed
from helpers.tf_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import count_accuracy, plot_recovered_graph


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
    dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.sem_type,
                               args.noise_scale, args.dataset_type, args.x_dim)
    _logger.info('Finished generating dataset')



    model = GAE(args.n, args.d, args.x_dim, args.seed, args.num_encoder_layers, args.num_decoder_layers,
                args.hidden_size, args.latent_dim, args.l1_graph_penalty, args.use_float64)
    model.print_summary(print_func=model.logger.info)

    trainer = ALTrainer(args.init_rho, args.rho_thres, args.h_thres, args.rho_multiply,
                        args.init_iter, args.learning_rate, args.h_tol,
                        args.early_stopping, args.early_stopping_thres)
    W_est = trainer.train(model, dataset.X, dataset.W, args.graph_thres,
                          args.max_iter, args.iter_step, output_dir)



    _logger.info('Finished training model')

    # Save raw recovered graph, ground truth and observational data after training
    np.save('{}/true_graph.npy'.format(output_dir), dataset.W)
    np.save('{}/observational_data.npy'.format(output_dir), dataset.X)
    np.save('{}/final_raw_recovered_graph.npy'.format(output_dir), W_est)

    # Plot raw recovered graph
    plot_recovered_graph(np.abs(W_est), np.abs(dataset.W),
                         save_name='{}/raw_recovered_graph.png'.format(output_dir))

    _logger.info('Filter by constant threshold and Normalize W_est & dataset.W')
    W_est = W_est / np.max(np.abs(W_est))    # Normalize
    W_truth_normalized = dataset.W / np.max(np.abs(dataset.W))    # Normalize

    # Plot thresholded recovered graph
    W_est[np.abs(W_est) < args.graph_thres] = 0    # Thresholding
    plot_recovered_graph(np.abs(W_est), np.abs(W_truth_normalized),
                         save_name='{}/thresholded_recovered_graph.png'.format(output_dir))
    results_thresholded = count_accuracy(dataset.W, W_est)
    _logger.info('Results after thresholding by {}: {}'.format(args.graph_thres, results_thresholded))


if __name__ == '__main__':
    main()
