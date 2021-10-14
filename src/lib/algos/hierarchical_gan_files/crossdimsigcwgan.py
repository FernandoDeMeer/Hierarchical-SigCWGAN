from dataclasses import dataclass

import torch
from sklearn.linear_model import LinearRegression
from torch import optim
import signatory

from src.lib.base import BaseAlgo, CrossDimAlgo, CrossDimConfig
from src.lib.augmentations import SignatureConfig
from src.lib.augmentations import augment_path_and_compute_signatures
from src.lib.utils import sample_indices, to_numpy


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


@dataclass
class CrossDimSigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_base(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_target(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)

    # def compute_sig(self, x):
    #     return  signatory.signature(x, 2, basepoint=False)



def calibrate_sigw1_metric_cross_dim(config, x_output, x_input, ):
    sigs_base = config.compute_sig_base(x_input)
    sigs_target = config.compute_sig_target(x_output)
    assert sigs_base.size(0) == sigs_target.size(0)
    X, Y = to_numpy(sigs_base), to_numpy(sigs_target)
    lm = LinearRegression()
    lm.fit(X, Y)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_output.device)
    return sigs_pred


def sample_sig_fake(G, sig_config, q, x_past_base, base_G,):
    # We repeat the inputs mc_size times for Monte Carlo Generation (through the base sigcwgan since the cross-dim doesn't add noise)
    x_past_base_mc = x_past_base.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    # We sample the base_dims with the base_G
    x_fake_base = base_G.sample(q,x_past_base_mc)
    # Now with our cross dim G we sample the target dims
    x_fake = G.sample_window(x_fake_base)
    # We unflatten the output and build x_fake
    x_fake = torch.reshape(x_fake, (x_fake.shape[0], q,-1))
    x_fake = torch.cat((x_fake_base,x_fake),dim= -1)
    # Now compute the signature of the generated window
    sigs_fake_future = sig_config.compute_sig_target(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past_base.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake


class CrossDimSigCWGAN(CrossDimAlgo):
    def __init__(
            self,
            cross_dim_config: CrossDimConfig,
            config: CrossDimSigCWGANConfig,
            x_input: torch.Tensor,
            x_output: torch.Tensor,
            x_input_base: torch.Tensor,
            base_sigcwgan: BaseAlgo
    ):
        super(CrossDimSigCWGAN, self).__init__(cross_dim_config, x_input, x_output, x_input_base, base_sigcwgan)
        self.cross_dim_config = cross_dim_config
        self.base_dims = cross_dim_config.base_dims
        self.sig_config = config
        self.mc_size = config.mc_size

        # The size of x_input and x_output should be p+q-1 and q respectively
        assert x_input.shape[1] == cross_dim_config.p and x_output.shape[1] == cross_dim_config.q
        self.sigs_pred = calibrate_sigw1_metric_cross_dim(config, x_output, x_input,)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=200, gamma=0.8)


    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices.long()].clone().to(self.device)
        x_input= self.x_input[random_indices.long()].clone().to(self.device)
        x_input_base = self.x_input_base[random_indices.long()].clone().to(self.device)
        return sigs_pred, x_input, x_input_base

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_input, x_input_base = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.sig_config,self.cross_dim_config.q,
                                               x_input_base,self.base_G,)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        # self.evaluate(x_fake) # This function logs the losses and metrics through training
