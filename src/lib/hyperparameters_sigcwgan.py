from src.lib.algos.sigcwgan_files.sigcwgan import SigCWGANConfig
from src.lib.augmentations import SignatureConfig, Scale, Cumsum, LeadLag,AddLags

SIGCWGAN_CONFIGS = dict(
    ETFS=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
    ),
    futures=SigCWGANConfig(
        mc_size=1000,
        sig_config_past=SignatureConfig(depth=2, augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),
        sig_config_future=SignatureConfig(depth=2,
                                          augmentations=tuple([ Scale(0.2), Cumsum(), AddLags(m=2), LeadLag()])),


    ),)
