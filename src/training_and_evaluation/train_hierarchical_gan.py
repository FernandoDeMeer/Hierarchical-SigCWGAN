import os
from os import path as pt

import numpy as np
import torch
import matplotlib.pyplot as plt
# os.chdir('/app')
from src.lib.algos.hierarchical_gan_files.hierarchical_gan import HierarchicalGAN
from src.lib.hyperparameters_hierarchicalgan import Clustering_Hierarchical_GAN_CONFIGS, CrossDim_SIGCWGAN_CONFIGS,Base_SIGCWGAN_CONFIGS
from src.lib.algos import ALGOS
from src.lib.base import BaseConfig,CrossDimConfig
from src.lib.data import get_data_cross_dim_sigcwgan, get_data_base_sigcwgan
from src.lib.plot import savefig, create_summary
from src.lib.utils import pickle_it

def get_base_sigcwgan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return Base_SIGCWGAN_CONFIGS[key]

def get_cross_dim_sigcwgan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return CrossDim_SIGCWGAN_CONFIGS[key]

def get_clust_hier_gan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return Clustering_Hierarchical_GAN_CONFIGS[key]

def get_algo(base_config, dataset, x_input,x_output):
    """
    x_input, x_output have to be the outputs of get_data_base_sigcwgan or get_data_cross_dim_sigcwgan to output the corresponding
    sigcwgan
    """
    algo_config = get_base_sigcwgan_config(dataset)

    algo = ALGOS['SigCWGAN'](x_input=x_input,x_output=x_output, config=algo_config, base_config=base_config)

    return algo

def get_cross_dim_algo(cross_dim_config, dataset, x_input,x_output, x_input_base, base_sigcwgan):
    """
    x_input, x_output have to be the outputs of get_data_base_sigcwgan or get_data_cross_dim_sigcwgan to output the corresponding
    sigcwgan
    """
    algo_config = get_cross_dim_sigcwgan_config(dataset)

    algo = ALGOS['CrossDimSigCWGAN'](x_input=x_input,x_output=x_output, config=algo_config,
                                     cross_dim_config=cross_dim_config, base_sigcwgan = base_sigcwgan,
                                     x_input_base = x_input_base)

    return algo

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_Base_Config(args):

    base_config = BaseConfig(
    device='cuda' if args.use_cuda else 'cpu',
    seed=args.initial_seed,
    batch_size=args.batch_size,
    hidden_dims=args.hidden_dims,
    p=args.p,
    q=args.q,
    total_steps=args.total_steps,)

    return base_config

def get_Cross_Dim_Config(args):

    cross_dim__config = CrossDimConfig(
    device='cuda' if args.use_cuda else 'cpu',
    seed=args.initial_seed,
    batch_size=args.batch_size,
    hidden_dims=args.hidden_dims,
    p=args.p,
    q=args.q,
    total_steps=args.total_steps,)

    return cross_dim__config



def run_base_sigcwgan(algo_id, base_config, base_dir, dataset, HierGAN):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s' % (algo_id, dataset))
    experiment_directory = pt.join(base_dir, dataset,'HierarchicalGAN', 'seed={}'.format(base_config.seed), 'BaseSigCWGAN')
    print('Saving to:' +experiment_directory)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    # Set seed for exact reproducibility of the experiments
    set_seed(base_config.seed)

    x_input,x_output = get_data_base_sigcwgan(dataset=dataset, p = base_config.p, q = base_config.q,
                                    clusters_last_added_array=HierGAN.clusters_last_added_array)
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
    savefig('dendrogram.png', experiment_directory)
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_input, pt.join(pt.dirname(experiment_directory), 'x_input.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cuda').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)

    return algo


def run_cross_dim_sigcwgan(algo_id, cross_dim_config, base_dir, dataset, base_dims, target_dims,cluster_id,base_sigcwgan):

    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s' % (algo_id, dataset))
    experiment_directory = pt.join(base_dir, dataset, 'HierarchicalGAN', 'seed={}'.format(cross_dim_config.seed), algo_id
                                   + '_cluster_{}'.format(cluster_id))
    print('Saving to:' + experiment_directory)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    # Set seed for exact reproducibility of the experiments
    set_seed(cross_dim_config.seed)
    # initialise dataset and algo

    x_input_target,x_output, x_input_base = get_data_cross_dim_sigcwgan(dataset=dataset, base_dims=base_dims,
                                                   target_dims=target_dims, p=cross_dim_config.p, q=cross_dim_config.q)
    x_input_target = x_input_target.to(cross_dim_config.device)
    x_output = x_output.to(cross_dim_config.device)
    x_input_base = x_input_base.to(cross_dim_config.device)
    algo = get_cross_dim_algo(cross_dim_config=cross_dim_config,
                    dataset=dataset,
                    x_input=x_input_target,
                    x_output= x_output,
                    x_input_base = x_input_base,
                    base_sigcwgan= base_sigcwgan
                    )
    # Train the algorithm
    algo.fit()
    # Pickle generator weights and training info.
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cuda').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log losses at the end of training
    algo.plot_losses()
    savefig('CrossDimSigW1loss.png', experiment_directory)


def main(args):
    print('Start of {} training. CUDA: {}'.format(args.algos, args.use_cuda))
    for dataset in args.datasets:
        for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
            base_config = get_Base_Config(args)
            cross_dim_config = get_Cross_Dim_Config(args)
            hier_gan_config = get_clust_hier_gan_config(dataset)
            HierGAN = HierarchicalGAN(hier_gan_config)
            # if not os.path.isdir(pt.join(args.base_dir, dataset,'seed={}'.format(base_config.seed), 'BaseSigCWGAN')):
            base_sigcwgan = run_base_sigcwgan(algo_id=args.algos[0], base_config=base_config, base_dir=args.base_dir,
                      dataset=args.datasets[0],
                      HierGAN = HierGAN)
            for key,value in HierGAN.models_to_train.items():
                if len(value)!=0: #some clusters may just have the base_dim
                    run_cross_dim_sigcwgan(algo_id='CrossDimSigCWGAN',cross_dim_config=cross_dim_config,base_dir=args.base_dir,
                                           dataset= dataset,
                                           base_dims= HierGAN.clusters_last_added_array,
                                           target_dims=np.array(value),
                                           cluster_id=key,
                                           base_sigcwgan = base_sigcwgan,)




