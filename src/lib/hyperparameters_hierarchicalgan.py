from src.lib.algos.hierarchical_gan_files.hierarchical_gan import clust_config
from src.lib.algos.sigcwgan_files.sigcwgan import SigCWGANConfig
from src.lib.algos.hierarchical_gan_files.crossdimsigcwgan import CrossDimSigCWGANConfig
from src.lib.augmentations import SignatureConfig, Scale, Cumsum, LeadLag, AddLags

Clustering_Hierarchical_GAN_CONFIGS = dict(
    futures= clust_config(
        mc_size=500,
        clust_method= 'Correlation',
        dataset= 'futures',
        experiment_dir='src/data/clustering',
        no_of_clusters = 5,
    ),
    futures_extended= clust_config(
        mc_size=500,
        clust_method= 'Correlation',
        dataset= 'futures_extended',
        experiment_dir='src/data/clustering',
        no_of_clusters = 15,
    ),)

Base_SIGCWGAN_CONFIGS = dict(
    futures=SigCWGANConfig(
        mc_size=500,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()]))
    ),
    futures_extended=SigCWGANConfig(
        mc_size=500,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()]))

    ),)

CrossDim_SIGCWGAN_CONFIGS = dict(
    futures=CrossDimSigCWGANConfig(
        mc_size=500,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),

    ),
    futures_extended=CrossDimSigCWGANConfig(
        mc_size=500,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),)
