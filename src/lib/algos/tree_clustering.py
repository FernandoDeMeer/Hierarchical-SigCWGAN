import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import signatory
import torch

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from src.lib.data import get_rawandpreprocessed_data,rolling_window
from src.lib.plot import plot_matrix


def get_corr_dist_matrix(dataset):

    pipeline, data_raw, data_preprocessed = get_rawandpreprocessed_data(data_type=dataset)
    corr_matrix = np.corrcoef(data_raw[0].numpy(),rowvar=False)
    Dist_matrix = np.sqrt((1/2)*(1-corr_matrix))

    return Dist_matrix

def get_signature_dist_matrix(dataset):
    """
        Given a multi-dimensional time series, this signature outputs a distance matrix with the average of euclidean
        distances between the signatures of rolling windows of size 5 of each of the dimensions of the time series.

        Params:

            prices: Dataframe or array
                The time series of prices from which to calculate the signature-based distance matrix.
    """
    pipeline, data_raw, data_preprocessed = get_rawandpreprocessed_data(data_type=dataset)

    windows = rolling_window(data_preprocessed, x_lag= 5, add_batch_dim= False)

    # Now calculate the signature for each dimension separately
    depth = 10
    signatures = torch.empty((windows.shape[0],depth,windows.shape[-1]))
    for i in range(windows.shape[-1]):

        signature = signatory.signature(torch.reshape(windows[:,:,i],(windows.shape[0],windows.shape[1],1)), depth=depth)
        signatures[:,:,i] = signature

    signature_dist_matrix = np.zeros((windows.shape[-1],windows.shape[-1]))

    for i in range(windows.shape[-1]):
        for j in range(i,windows.shape[-1]):
            signature_dist_matrix[i,j] = np.average(np.linalg.norm((signatures[:,:,i]-signatures[:,:,j]),ord=1))

    signature_dist_matrix = signature_dist_matrix + np.transpose(signature_dist_matrix)-np.diag(signature_dist_matrix)

    return signature_dist_matrix


def get_linkage(dataset,Dist_matrix,experiment_directory,method='complete'):
    try:
        multi_asset_data = pd.read_csv('src/data/{}.csv'.format(dataset),index_col=0,header=0)
    except:
        multi_asset_data = pd.read_csv('../data/{}.csv'.format(dataset),index_col=0,header=0)

    myComplete = linkage(Dist_matrix,method=method)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=40)
    R = dendrogram(
        myComplete,
        no_plot=True,
    )

    # print("values passed to leaf_label_func\nleaves : ", R["leaves"])
    # labels = Dist_matrix.columns(R["leaves"])
    dendrogram(
        myComplete,
        labels=multi_asset_data.columns,
        leaf_rotation=90.,
        leaf_font_size=8,
    )

    # print("values passed to leaf_label_func\nleaves : ", R["leaves"])

    # plt.savefig(experiment_directory + '/completetree.png')

    # pd.DataFrame(myComplete).to_csv(experiment_directory + '/CompleteLinkage_py.csv')

    return R, myComplete

def get_linkage_from_dataset(dataset,experiment_directory,method):

    if method == 'Correlation':
        dist_matrix = get_corr_dist_matrix(dataset)
    elif method == 'Signature':
        dist_matrix = get_signature_dist_matrix(dataset)
    try:
        f = open(experiment_directory + '/{}_matrix.png'.format(method))
        f.close()
    except FileNotFoundError:

        plot_matrix(dataset,dist_matrix,experiment_directory,method)


    R, complete_linkage = get_linkage(dataset,dist_matrix,experiment_directory,'complete')

    return R, complete_linkage

def get_clusters(linkage,n_of_clusters,experiment_directory):

    clusters = fcluster(Z=linkage,t=n_of_clusters,criterion='maxclust')
    # pd.DataFrame(clusters).to_csv(experiment_directory + '/Clusters.csv')

    return clusters

def get_last_added(linkage,clusters):
    """
    Given a set of clusters based on a linkage matrix, this function outputs the last element added to each of the clusters.

    Params:

        linkage: scipy.cluster.hierarchy.linkage array

            The linkage matrix.

        clusters: scipy.cluster.hierarchy.fcluster array

            The cluster vector, each original observation is indexed by its cluster number.
    """

    n_of_clusters = np.max(clusters)
    clusters_dict = {}
    for i in range(1,n_of_clusters+1):
        clusters_dict[str(i)] = np.where(clusters==i)
    clusters_representatives_dict = {}
    for i in range(1,n_of_clusters+1):
        for j in range(1,np.shape(linkage)[0] + 1):
            if np.any(np.in1d(clusters_dict[str(i)], linkage[-j,:2])):
                cluster = clusters_dict[str(i)]
                for k in range(np.size(cluster)):
                    if np.in1d(clusters_dict[str(i)], linkage[-j,:2])[k]:
                        clusters_representatives_dict[str(i)] = cluster[0][k]
                        break
                break
    clusters_representative_array = []
    for key, values in clusters_representatives_dict.items():

        clusters_representative_array.append(values)
    clusters_representative_array = np.array((clusters_representative_array))
    return clusters_dict,clusters_representatives_dict,clusters_representative_array

def get_clusters_from_dataset(dataset,experiment_directory,method,no_of_clusters):
    R, complete_linkage = get_linkage_from_dataset(dataset,experiment_directory,method)
    clusters = get_clusters(linkage= complete_linkage, n_of_clusters=no_of_clusters, experiment_directory= experiment_directory)
    clusters_dict,clusters_representatives_dict,clusters_representatives_array = get_last_added(linkage=complete_linkage,clusters=clusters)
    # pd.DataFrame(clusters_representatives_array).to_csv(experiment_directory + "/clusters_representatives.csv")

    return clusters, clusters_dict, clusters_representatives_dict,clusters_representatives_array


if __name__ == '__main__':
    test_directory = '../../data/clustering'

    # clusters, clusters_dict, clusters_last_added_dict,clusters_last_added_array = get_clusters_from_prices(prices_2020,test_directory)
    # plot_distance_matrix(prices_2020,signature_dist_matrix,test_directory)
    # dist_matrix = get_signature_dist_matrix('futures')
    # plot_matrix('futures',dist_matrix,test_directory,'Signature')
    # R, complete_linkage = get_linkage('futures',dist_matrix,test_directory,'complete')
    clusters, clusters_dict, clusters_last_added_dict,clusters_last_added_array = get_clusters_from_dataset('futures',test_directory,'Correlation')
