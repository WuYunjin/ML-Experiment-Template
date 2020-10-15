import sys
import yaml
import argparse


def load_yaml_config(path, skip_lines=0):
    with open(path, 'r') as infile:
        for i in range(skip_lines):
            # Skip some lines (e.g., namespace at the first line)
            _ = infile.readline()

        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    parser = argparse.ArgumentParser()

    ##### General settings #####
    parser.add_argument('--seed',
                        type=int,
                        default=1230,
                        help='Random seed')
    
    ##### Dataset settings #####
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Number of batch size')


    ##### Model settings #####
    parser.add_argument('--num_hidden',
                        type=int,
                        default=100,
                        help='Hidden size for NN layers')


    parser.add_argument('--z_dim',
                        type=int,
                        default=32,
                        help='the dimension of latent variable z')


    parser.add_argument('--input_dim',
                        type=int,
                        default=28*28,
                        help='the size of input data')
                        
    
    ##### Training settings #####
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for Adam optimizer')


    parser.add_argument('--num_epochs',
                        type=float,
                        default=2,
                        help='Number of epochs')



    ##### Other settings #####

    return parser.parse_args(args=sys.argv[1:])
