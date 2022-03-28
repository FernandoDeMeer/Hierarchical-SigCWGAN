import os
from os import path as pt
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from src.lib.algos.tree_clustering import get_clusters_from_dataset
from src.lib.data import get_data_sigcwgan,rolling_window, rolling_window_non_overlapping, get_rawandpreprocessed_data
from src.lib.algos.sigcwgan_files.sigcwgan import calibrate_sigw1_metric,sigcwgan_loss,sample_sig_fake
from src.lib.algos.hierarchical_gan_files.hierarchical_gan import HierarchicalGAN
from src.lib.hyperparameters_hierarchicalgan import Clustering_Hierarchical_GAN_CONFIGS,Base_SIGCWGAN_CONFIGS,CrossDim_SIGCWGAN_CONFIGS
from src.lib.hyperparameters_sigcwgan import SIGCWGAN_CONFIGS
from src.lib.base import BaseConfig,CrossDimConfig
from src.lib.arfnn import SimpleGenerator,CrossDimGenerator
from src.lib.utils import load_pickle, to_numpy
from src.lib.base import is_multivariate
from src.lib.plot import plot_summary, compare_cross_corr
from src.lib.evaluate_utils import rolling_window_base_dim,\
    compute_predictive_score,compute_test_metrics, sample_sig_fake_hiergan, returns_to_prices
from src.lib.discriminative_score import discriminative_score


def get_clust_hier_gan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return Clustering_Hierarchical_GAN_CONFIGS[key]

def get_base_sigcwgan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return Base_SIGCWGAN_CONFIGS[key]

def get_crossdim_sigcwgan_config(dataset):
    """ Get the algorithms parameters. """
    key = dataset

    return CrossDim_SIGCWGAN_CONFIGS[key]

def get_top_dirs(path):
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]

def get_original_dataset(dataset):
    data_function_name = 'get_{}_dataset'.format(dataset)
    try:
        from src import lib
        data_function = getattr(lib.data, data_function_name)
    except:
        pass
    try:
        pipeline, x_real_raw, x_real = data_function()
    except:
        raise NotImplementedError('Dataset %s not valid' % dataset)
    return pipeline, x_real_raw, x_real


def generate_series_hierarchical_gan(base_dir, use_cuda, datasets, series_to_generate, days_to_generate):
    msg = 'Generating series on GPU.' if use_cuda else 'Generating series on CPU.'
    print(msg)

    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if dataset_dir not in datasets:
            continue
        for experiment in os.listdir(dataset_path):
            if experiment == 'HierarchicalGAN':
                cross_dim_config = CrossDimConfig(device='cpu')
                experiment_path = os.path.join(dataset_path, experiment)
                hier_gan_config = get_clust_hier_gan_config(dataset_dir)
                hier_gan_config.experiment_dir='../data/clustering'
                HierGAN = HierarchicalGAN(hier_gan_config)


                for seed in get_top_dirs(experiment_path):
                    experiment_path = os.path.join(experiment_path, seed)
                    for algo in get_top_dirs(experiment_path):

                        algo_path = os.path.join(experiment_path, algo)

                        if algo == 'BaseSigCWGAN':

                            gen_rtns,current_generated_dims = load_weights_and_generate_base_sigcwgan(experiment_dir = algo_path,
                                                                                  dataset=dataset_dir , use_cuda=use_cuda,
                                                                                  series_to_generate = series_to_generate,
                                                                                  days_to_generate= days_to_generate,
                                                                                  cross_dim_config= cross_dim_config)
                        else:

                            [cluster_number] = [int(s) for s in algo.split('_') if s.isdigit()]
                            if HierGAN.models_to_train[str(cluster_number)]!=[]:
                                gen_rtns,current_generated_dims =load_weights_and_generate_cross_dim_sigcwgan(
                                                                             experiment_dir = algo_path,
                                                                             dataset=dataset_dir , use_cuda=use_cuda,
                                                                             cross_dim_config= cross_dim_config,
                                                                             gen_rtns= gen_rtns,
                                                                             base_dims = HierGAN.clusters_last_added_array,
                                                                             target_dims=HierGAN.models_to_train[str(cluster_number)],
                                                                             current_generated_dims=current_generated_dims)

                    pipeline, x_real_raw, x_real = get_original_dataset(dataset=dataset_dir)
                    gen_rtns = to_numpy(pipeline.inverse_transform(gen_rtns))
                    np.save(experiment_path + '/generated_scenarios.npy',gen_rtns,)

    return gen_rtns


def load_weights_and_generate_cross_dim_sigcwgan(experiment_dir, dataset, use_cuda,cross_dim_config,
                                                 gen_rtns, base_dims,
                                                 target_dims, current_generated_dims):
    """
    Args:
        experiment_dir: directory of the experiment.
        dataset: dataset to be generated.
        use_cuda: GPU/CPU generation.
        series_to_generate: Amount of scenarios to generate
        days_to_generate:
    Returns:
        Generated series .npy
    """
    # the seed has already been set by the BaseSigCWGAN

    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # shorthands
    p, q = cross_dim_config.p, cross_dim_config.q
    # ----------------------------------------------
    # Get the base_dim
    # ----------------------------------------------

    x_past = rolling_window_base_dim(torch.from_numpy(np.array(gen_rtns[:, :, base_dims])), p)
    dim = np.array(target_dims).size

    # ----------------------------------------------
    # Load generator weights and hyperparameters
    # ----------------------------------------------
    G_weights = load_pickle(os.path.join(experiment_dir, 'G_weights.torch'))
    G = CrossDimGenerator(p*x_past.shape[-1], q*dim, 3 * (50,),).to(device)
    G.load_state_dict(G_weights)
    # ----------------------------------------------
    # Generate paths
    # ----------------------------------------------
    for scenario in range(x_past.shape[0]):
        with torch.no_grad():
            input_windows = x_past[scenario]
            output = G.sample_window(input_windows)
            output = torch.reshape(output, (output.shape[0],q, -1))
            output = torch.cat((list(output)),dim = 0)
            gen_rtns[scenario, :, target_dims] = output


    # The pipeline is taken from the original dataset, so we need to plug the generated returns into the gen_rtns_base_dim
    pipeline, x_real_raw, x_real = get_original_dataset(dataset)

    # Add to cumulative matrix of returns (we denormalize the generated returns before plotting)

    gen_rtns_plot = to_numpy(pipeline.inverse_transform(gen_rtns))

    # SigCWGAN generates absolute returns, we now calculate prices
    current_generated_dims = np.append(current_generated_dims, target_dims)

    gen_rtns_plot = gen_rtns_plot[:, :, current_generated_dims]
    gen_prices_plot = returns_to_prices(gen_rtns_plot)

    # Now plot the iteratively generated scenario

    fig, ax1 = plt.subplots(1, 1)
    experiment_dir = Path(experiment_dir).parent
    experiment_dir = experiment_dir.__str__()
    if not pt.exists(experiment_dir + '/BaseSigCWGAN/graphs'):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_dir + '/BaseSigCWGAN/graphs')

    for i in range(25):
        plt.plot((gen_prices_plot[i,:,:]))
        plt.savefig(experiment_dir + '/BaseSigCWGAN/graphs/Scenario_{}_{}_dimension_{}.png'.format(i, current_generated_dims.size, target_dims))
        if current_generated_dims.size == gen_prices_plot.shape[-1]:
            plt.clf()
    return gen_rtns, current_generated_dims

def load_weights_and_generate_base_sigcwgan(experiment_dir, dataset, use_cuda, cross_dim_config, series_to_generate,days_to_generate):
    """
    Args:
        experiment_dir: directory of the experiment.
        dataset: dataset to be generated.
        use_cuda: GPU/CPU generation.
        series_to_generate: Amount of scenarios to generate
        days_to_generate: Amount of days to generate for each scenario
    Returns:
        Generated series .npy
    """
    torch.random.manual_seed(0)
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # shorthands
    base_config = BaseConfig(device=device)
    p, q = base_config.p, base_config.q
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = load_pickle(os.path.join(os.path.dirname(experiment_dir), 'x_input.torch')).to(device) # This is x_real for the BaseSIGCWGAN, not the entire original dataset.
    x_past = x_real[:, :p]
    dim = x_real.shape[-1]
    # ----------------------------------------------
    # Load generator weights and hyperparameters
    # ----------------------------------------------
    G_weights = load_pickle(os.path.join(experiment_dir, 'G_weights.torch'))
    G = SimpleGenerator(dim * p, dim, 3 * (50,), dim).to(device)
    G.load_state_dict(G_weights)
    # Generate long paths, first get conditional inputs
    # ----------------------------------------------
    cond_info = torch.reshape(x_past[0],shape=(1,-1,x_past.shape[-1]))
    for i in range(series_to_generate-1):
        idx = np.random.randint(0,x_past.shape[0])
        cond_info = torch.cat((cond_info,torch.reshape(x_past[idx],shape=(1,-1,x_past.shape[-1]))),dim=0)
    # ----------------------------------------------

    # ----------------------------------------------
    # Generate long paths
    # ----------------------------------------------
    if days_to_generate % cross_dim_config.p != 0:
        days_to_generate_new = cross_dim_config.p*(int(days_to_generate/cross_dim_config.p)+1)
    else:
        days_to_generate_new = days_to_generate


    with torch.no_grad():
        x_fake = G.sample(days_to_generate_new, cond_info)

    # We need to learn which dimensions the BaseSigCWGAN is
    # generating, so we call the clustering functions

    clusters, clusters_dict, \
    clusters_last_added_dict,clusters_last_added_array = get_clusters_from_dataset(dataset,experiment_dir,'Correlation',no_of_clusters=dim)
    gen_rtns = torch.zeros((series_to_generate,days_to_generate_new, clusters.size))

    for dim in range(clusters_last_added_array.size):
        gen_rtns[:,:,clusters_last_added_array[dim]] = x_fake[:,:,dim]

    # pipeline, x_real_raw, x_real = get_original_dataset(dataset)
    # gen_rtns = to_numpy(pipeline.inverse_transform(gen_rtns))

    # # We go back to our original format for plotting purposes
    #
    # gen_rtns_plot = gen_rtns[:,:,clusters_last_added_array]

    # # SigCWGAN generates absolute returns, we transform back to prices to plot

    # gen_prices = returns_to_prices(gen_rtns_plot)

    # fig, ax1 = plt.subplots(1, 1)
    # if not pt.exists(experiment_dir + '/graphs'):
    #     # if the experiment directory does not exist we create the directory
    #     os.makedirs(experiment_dir + '/graphs')
    #
    # for i in range(gen_prices.shape[0]):
    #     plt.plot((gen_prices[i,:,:]))

    #     plt.savefig(experiment_dir + '/graphs/Scenario_{}_base_dims.png'.format(i))
    #     plt.clf()
    #     # plt.show()

    return gen_rtns,clusters_last_added_array



def evaluate_generator(model_name, seed, experiment_dir, dataset, use_cuda=True):
    """
    Args:
        model_name:
        experiment_dir:
        dataset:
        use_cuda:
    Returns:
    """
    torch.random.manual_seed(0)
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    experiment_summary = dict()
    experiment_summary['model_id'] = model_name
    experiment_summary['seed'] = seed

    sig_config = SIGCWGAN_CONFIGS[dataset]

    # shorthands
    base_config = BaseConfig(device=device)
    p, q = base_config.p, base_config.q
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = get_data_sigcwgan(dataset,p,q)
    x_past, x_future = x_real[:, :p], x_real[:, p:p + q]
    dim = x_real.shape[-1]
    # ----------------------------------------------
    # Load generator weights and hyperparameters
    # ----------------------------------------------
    G_weights = load_pickle(os.path.join(experiment_dir, 'G_weights.torch'))
    G = SimpleGenerator(dim * p, dim, 3 * (50,), dim).to(device)
    G.load_state_dict(G_weights)

    pipeline, x_real_raw, x_real = get_rawandpreprocessed_data(dataset)
    real_data = rolling_window_non_overlapping(x_real_raw[0], 25)

    # Generate as many scenarios with the SigCWGAN
    gen_data = G.sample(25, real_data[:,:p]).detach().numpy()
    gen_data = pipeline.inverse_transform(torch.Tensor(gen_data))
    # gen_data = rolling_window_non_overlapping(torch.Tensor(gen_data[0]), 25)

    real_train_data = real_data[:200]
    real_test_data = real_data[200:]
    gen_train_data = gen_data[:200]
    gen_test_data = gen_data[200:]

    # ----------------------------------------------
    # Compute discriminative score
    # ----------------------------------------------
    disc_score = discriminative_score(real_train_data, gen_train_data,real_test_data, gen_test_data)
    experiment_summary['discriminative_score'] = disc_score[0]

    # ----------------------------------------------
    # Compute predictive score - TSTR (train on synthetic, test on real)
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = G.sample(1, x_past)
    predict_score_dict = compute_predictive_score(x_past, x_future, x_fake)
    experiment_summary.update(predict_score_dict)
    # ----------------------------------------------
    # Compute metrics and scores of the unconditional distribution.
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = G.sample(q, x_past)
    test_metrics_dict = compute_test_metrics(x_fake, x_real)
    experiment_summary.update(test_metrics_dict)

    # ----------------------------------------------
    # Compute Sig-W_1 distance.
    # ----------------------------------------------
    sigs_pred = calibrate_sigw1_metric(sig_config, x_future, x_past)
    # generate fake paths
    sigs_conditional = list()
    with torch.no_grad():
        steps = 100
        size = x_past.size(0) // steps
        for i in range(steps):
            x_past_sample = x_past[i * size:(i + 1) * size] if i < (steps - 1) else x_past[i * size:]
            sigs_fake_ce = sample_sig_fake(G, q, sig_config, x_past_sample)[0]
            sigs_conditional.append(sigs_fake_ce)
        sigs_conditional = torch.cat(sigs_conditional, dim=0)
        sig_w1_metric = sigcwgan_loss(sigs_pred, sigs_conditional)
    experiment_summary['sig_w1_metric'] = sig_w1_metric.item()
    # ----------------------------------------------
    # Create the relevant summary plots.
    # ----------------------------------------------
    with torch.no_grad():
        _x_past = x_past.clone()
        x_fake_future = G.sample(q, _x_past)
        plot_summary(x_fake=x_fake_future, x_real=x_real, max_lag=q)
    plt.savefig(os.path.join(experiment_dir, 'summary.png'))
    plt.close()
    if is_multivariate(x_real):
        compare_cross_corr(x_fake=x_fake_future, x_real=x_real,GAN_type='SigCWGAN')
        plt.savefig(os.path.join(experiment_dir, 'cross_correl.png'))
        plt.close()
    # ----------------------------------------------
    # Generate long paths
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = G.sample(8000, x_past[0:1])
    plot_summary(x_fake=x_fake, x_real=x_real, max_lag=q)
    plt.savefig(os.path.join(experiment_dir, 'summary_long.png'))
    plt.close()
    plt.plot(to_numpy(x_fake[0, :1000]))
    plt.savefig(os.path.join(experiment_dir, 'long_path.png'))
    plt.close()
    return experiment_summary

def evaluate_hierarchical_generator(model_name, seed, experiment_dir, dataset, use_cuda=True):
    """
    Args:
        model_name:
        experiment_dir:
        dataset:
        use_cuda:
    Returns:
    """
    torch.random.manual_seed(0)
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    experiment_summary = dict()
    experiment_summary['model_id'] = model_name
    experiment_summary['seed'] = seed

    # shorthands
    base_sigcwgan_config = get_base_sigcwgan_config(dataset)
    base_config = BaseConfig(device=device)
    p, q = base_config.p, base_config.q
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = get_data_sigcwgan(dataset,p,q)
    x_past, x_future = x_real[:, :p], x_real[:, p:p + q]
    hier_gan_config = get_clust_hier_gan_config(dataset)
    hier_gan_config.experiment_dir='../data/clustering'
    HierGAN = HierarchicalGAN(hier_gan_config)

    pipeline, x_real_raw, x_real = get_rawandpreprocessed_data(dataset)
    real_data = rolling_window_non_overlapping(x_real_raw[0], 25)


    # Generate one scenario of the same size as x_real_raw
    gen_data = sample_hierarchical_gan(HierGAN=HierGAN,experiment_path=experiment_dir,
                                       dataset=dataset,days_to_generate=25,x_past=real_data[:,:p])
    gen_data = pipeline.inverse_transform(torch.Tensor(gen_data))

    real_train_data = real_data[:200]
    real_test_data = real_data[200:]
    gen_train_data = gen_data[:200]
    gen_test_data = gen_data[200:]

    # ----------------------------------------------
    # Compute discriminative score
    # ----------------------------------------------
    disc_score = discriminative_score(real_train_data, gen_train_data,real_test_data, gen_test_data)
    experiment_summary['discriminative_score'] = disc_score[0]

    # ----------------------------------------------
    # Compute predictive score - TSTR (train on synthetic, test on real)
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = sample_hierarchical_gan(HierGAN=HierGAN,experiment_path=experiment_dir,
                                         dataset=dataset,days_to_generate=1,x_past=x_past)
    predict_score_dict = compute_predictive_score(x_past, x_future, x_fake)
    experiment_summary.update(predict_score_dict)
    # ----------------------------------------------
    # Compute metrics and scores of the unconditional distribution.
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = sample_hierarchical_gan(HierGAN=HierGAN,experiment_path=experiment_dir,
                                                          dataset=dataset,days_to_generate=q,x_past=x_past)

    test_metrics_dict = compute_test_metrics(torch.from_numpy(x_fake), x_real)
    experiment_summary.update(test_metrics_dict)
    # ----------------------------------------------
    # Compute Sig-W_1 distance.
    # ----------------------------------------------

    sigs_pred = calibrate_sigw1_metric(base_sigcwgan_config, x_future, x_past)
    # generate fake paths
    sigs_conditional = list()
    with torch.no_grad():
        steps = 100
        size = x_past.size(0) // steps
        for i in range(steps):
            x_past_sample = x_past[i * size:(i + 1) * size] if i < (steps - 1) else x_past[i * size:]
            x_fake = sample_hierarchical_gan(HierGAN=HierGAN,experiment_path=experiment_dir,
                                             dataset=dataset,days_to_generate=q,x_past=x_past_sample)
            sigs_fake_ce = sample_sig_fake_hiergan(base_sigcwgan_config, x_past_sample,torch.from_numpy(x_fake))[0]
            sigs_conditional.append(sigs_fake_ce)
        sigs_conditional = torch.cat(sigs_conditional, dim=0)
        sig_w1_metric = sigcwgan_loss(sigs_pred, sigs_conditional)
    experiment_summary['sig_w1_metric'] = sig_w1_metric.item()


    # ----------------------------------------------
    # Create the relevant summary plots.
    # ----------------------------------------------
    with torch.no_grad():
        _x_past = x_past.clone()
        x_fake_future = sample_hierarchical_gan(HierGAN=HierGAN,experiment_path=experiment_dir,
                                         dataset=dataset,days_to_generate=q,x_past=_x_past)
        plot_summary(x_fake=torch.from_numpy(x_fake_future), x_real=x_real, max_lag=q)
    plt.savefig(os.path.join(experiment_dir, 'summary.png'))
    plt.close()
    if is_multivariate(x_real):
        compare_cross_corr(x_fake=torch.from_numpy(x_fake_future), x_real=x_real,GAN_type= 'Hier-SigCWGAN')
        plt.savefig(os.path.join(experiment_dir, 'cross_correl.png'))
        plt.close()


    return experiment_summary

def sample_hierarchical_gan(HierGAN,experiment_path, dataset, days_to_generate, x_past):

    for algo in get_top_dirs(experiment_path):

        algo_path = os.path.join(experiment_path, algo)

        if algo == 'BaseSigCWGAN':
            # shorthands
            base_config = BaseConfig(device='cpu')
            p, q = base_config.p, base_config.q
            dim = len(HierGAN.models_to_train)

            # shorthands
            cross_dim_config = CrossDimConfig(device='cpu')
            p_cross_dim = cross_dim_config.p
            q_cross_dim = cross_dim_config.q


        # Load generator weights and hyperparameters
            G_weights = load_pickle(os.path.join(algo_path, 'G_weights.torch'))
            base_G = SimpleGenerator(dim * p, dim, 3 * (50,), dim).to('cpu')
            base_G.load_state_dict(G_weights)

            # Generate
            x_input = x_past[:,:,HierGAN.clusters_last_added_array]
            if days_to_generate % p_cross_dim != 0:
                days_to_generate_new = p_cross_dim*(int(days_to_generate/p_cross_dim)+1)
            else:
                days_to_generate_new = days_to_generate

            with torch.no_grad():
                x_fake = base_G.sample( days_to_generate_new, x_input)

            # The pipeline is taken from the original dataset, so we need to learn which dimensions the BaseSigCWGAN is
            # generating, so we call the clustering functions
            pipeline, x_real_raw, x_real = get_original_dataset(dataset)

            gen_rtns = torch.zeros((x_input.shape[0],days_to_generate_new,x_real.shape[-1]))
            clusters, clusters_dict, \
            clusters_last_added_dict,clusters_last_added_array = get_clusters_from_dataset(dataset,experiment_path,'Correlation',no_of_clusters=dim)

            for dim in range(clusters_last_added_array.size):
                gen_rtns[:,:,clusters_last_added_array[dim]] = x_fake[:,:,dim]

        else:
            [cluster_number] = [int(s) for s in algo.split('_') if s.isdigit()]
            if HierGAN.models_to_train[str(cluster_number)]!=[]:
                base_dims = np.array(HierGAN.clusters_last_added_array)
                target_dims = np.array(HierGAN.models_to_train[str(cluster_number)])

                # Load generator weights and hyperparameters
                G_weights = load_pickle(os.path.join(algo_path, 'G_weights.torch'))
                G = CrossDimGenerator(p_cross_dim*(base_dims.size), q_cross_dim*(target_dims.size), 3 * (50,),).to('cpu')
                G.load_state_dict(G_weights)

                # Generate paths
                x_fake_base = gen_rtns[:,:,HierGAN.clusters_last_added_array]
                x_fake_base = rolling_window_base_dim(x_fake_base, p_cross_dim)
                for scenario in range(x_fake_base.shape[0]):
                    with torch.no_grad():
                        input_windows = x_fake_base[scenario]
                        output = G.sample_window(input_windows)
                        output = torch.reshape(output, (output.shape[0],-1, target_dims.size))
                        gen_rtns[scenario, :, target_dims] = output.reshape((-1,output.shape[-1]))

    # We denormalize the generated returns before returning them
    # pipeline, x_real_raw, x_real = get_original_dataset(dataset)
    # generated_data_rtns = to_numpy(pipeline.inverse_transform(gen_rtns))
    generated_data_rtns = to_numpy(gen_rtns[:,:days_to_generate,:])

    return generated_data_rtns

def evaluate_benchmarks(base_dir, datasets, use_cuda=False):
    msg = 'Running evalution on GPU.' if use_cuda else 'Running evalution on CPU.'
    print(msg)
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if dataset_dir not in datasets:
            continue
        for model_name in os.listdir(dataset_path):
            df = pd.DataFrame(columns=[])
            experiment_path = os.path.join(dataset_path, model_name)
            for seed_dir in get_top_dirs(experiment_path):
                seed_path = os.path.join(experiment_path, seed_dir)
                print(dataset_dir, model_name,)
                # evaluate the model
                if model_name == 'SigCWGAN':
                    pass
                    experiment_summary = evaluate_generator(
                        model_name=model_name,
                        seed=seed_dir.split('_')[-1],
                        experiment_dir=seed_path,
                        dataset=dataset_dir,
                        use_cuda=use_cuda
                    )
                    df = df.append(experiment_summary, ignore_index=True, )
                elif model_name == 'HierarchicalGAN':
                    experiment_summary = evaluate_hierarchical_generator(
                        model_name=model_name,
                        seed=seed_dir.split('_')[-1],
                        experiment_dir=seed_path,
                        dataset=dataset_dir,
                        use_cuda=use_cuda
                    )
                    df = df.append(experiment_summary, ignore_index=True,)

            df_dst_path = os.path.join(base_dir, dataset_dir, model_name, 'summary.csv')
            df.to_csv(df_dst_path, decimal='.', sep=',', float_format='%.5f', index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Turn cuda off / on during evaluation.')
    parser.add_argument('-base_dir', default='../generated_data', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-datasets', default=['futures', ], nargs="+")
    parser.add_argument('-series_to_generate', type=int, default=1000)
    parser.add_argument('-days_to_generate', type=int, default=500)



    args = parser.parse_args()

    # generate_series_hierarchical_gan(base_dir=args.base_dir, use_cuda=args.use_cuda, datasets=args.datasets,
    #                                  series_to_generate=args.series_to_generate, days_to_generate=args.days_to_generate)

    evaluate_benchmarks( base_dir = args.base_dir, datasets = args.datasets, use_cuda=False)

