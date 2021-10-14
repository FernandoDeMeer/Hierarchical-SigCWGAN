from dataclasses import dataclass
import numpy as np

from src.lib.algos.tree_clustering import get_clusters_from_dataset


@dataclass
class clust_config:
    mc_size: int
    clust_method : str
    dataset: str
    experiment_dir : str
    no_of_clusters : int

    def get_all_models_to_train(self,clusters,clusters_last_added_array):
        """
        Given a set of clusters based on the linkage matrix, this function outputs
        a list of all the models to train in the ClustHierGAN, each asset that was not
        added last to its cluster, will be generated through a CrossDimSigCWGAN.

        Params:

            clusters: ndarray

                Array that assigns to each asset a cluster i.e. if clusters[0]=1 that means the asset in position 0
                belongs to the first cluster.

            clusters_last_added_array: ndarray

                Array which contains the index of the last added asset to each cluster i.e. if clusters_last_added_array[0] = 1
                then the asset in position 1 was added last to cluster 0.

        Output:

            models_to_train: dict

                Dictionary where each entry is indexed by the cluster number and the values are the indexes of assets
                included in the cluster and not in the base_dims
        """

        models_to_train = {}
        # clusters is an array where the value of each position is that asset's cluster
        for i in range(np.size(clusters)):
            current_cluster = clusters[i]
            if not str(current_cluster) in models_to_train:
                models_to_train[str(current_cluster)] = []
            if not np.isin(i, clusters_last_added_array):
                models_to_train[str(current_cluster)].append(i)

        return models_to_train



class HierarchicalGAN():

    def __init__(self,
                 config: clust_config, ):
        self.clust_config = config
        self.mc_size = config.mc_size


        clusters, clusters_dict, \
        clusters_last_added_dict,\
        clusters_last_added_array = get_clusters_from_dataset(config.dataset,config.experiment_dir,
                                                              config.clust_method,config.no_of_clusters)

        self.clusters = clusters
        self.clusters_dict = clusters_dict
        self.clusters_last_added_dict = clusters_last_added_dict
        self.clusters_last_added_array = clusters_last_added_array

        models_to_train = self.clust_config.get_all_models_to_train(clusters,clusters_last_added_array)

        self.models_to_train = models_to_train








