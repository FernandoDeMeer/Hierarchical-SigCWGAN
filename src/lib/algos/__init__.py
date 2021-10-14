from src.lib.algos.sigcwgan_files.sigcwgan import SigCWGAN
from src.lib.algos.hierarchical_gan_files.hierarchical_gan import HierarchicalGAN
from src.lib.algos.hierarchical_gan_files.crossdimsigcwgan import CrossDimSigCWGAN

ALGOS = dict(SigCWGAN=SigCWGAN,ClustHierarchicalGAN = HierarchicalGAN,CrossDimSigCWGAN=CrossDimSigCWGAN)
