import os
from os import path as pt

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.lib.hyperparameters_sigcwgan import SIGCWGAN_CONFIGS
from src.lib.algos import ALGOS
from src.lib.base import BaseConfig
from src.lib.data import get_data_sigcwgan
from src.lib.plot import savefig, create_summary
from src.lib.utils import pickle_it


def get_sigcwgan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return SIGCWGAN_CONFIGS[key]

def get_algo(base_config, dataset, x_input,x_output):

    algo_config = get_sigcwgan_config(dataset,)
    algo = ALGOS['SigCWGAN'](x_input=x_input,x_output=x_output, config=algo_config, base_config=base_config)

    return algo

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_sigcwgan(algo_id, base_config, base_dir, dataset,):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s' % (algo_id, dataset))
    experiment_directory = pt.join(base_dir, dataset,'SigCWGAN', 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    # Set seed for exact reproducibility of the experiments
    set_seed(base_config.seed)
    # initialise dataset and algo
    x_input,x_output = get_data_sigcwgan(data_type=dataset,p=base_config.p,q=base_config.q)
    x_input = x_input.to(base_config.device)
    x_output = x_output.to(base_config.device)

    algo = get_algo(base_config=base_config, dataset=dataset, x_input=x_input, x_output = x_output)
    # Train the algorithm
    algo.fit()
    # create summary
    create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_output)
    savefig('summary.png', experiment_directory)
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, 8000, x_output, one=True)
    savefig('summary_long.png', experiment_directory)
    plt.plot(x_fake.cpu().numpy()[0, :2000])
    savefig('long_path.png', experiment_directory)
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_input, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cuda').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)

def get_dataset_configuration(dataset):
    if dataset == 'ETFS':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('Universe1',)])
    elif dataset == 'futures':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('Universe1',)])
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator

def main(args):
    print('Start of {} training. CUDA: {}'.format(args.algos, args.use_cuda))
    for dataset in args.datasets:
        for algo_id in args.algos:
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                base_config = BaseConfig(
                    device='cuda' if args.use_cuda else 'cpu',
                    seed=seed,
                    batch_size=args.batch_size,
                    hidden_dims=args.hidden_dims,
                    p=args.p,
                    q=args.q,
                    total_steps=args.total_steps,)
                generator = get_dataset_configuration(dataset)
                for spec, data_params in generator:
                    run_sigcwgan(
                        algo_id=algo_id,
                        base_config=base_config,
                        dataset=dataset,
                        base_dir=args.base_dir
                        ,)



