import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *

import clip


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.diffusion = KarrasPoint(
            net = PointwiseNet(),
        )

    def decode(self, context, y, num_points, ret_traj=False):
        samples = None
        for x in self.diffusion.sample(num_points, context, y, ret_traj=ret_traj):
            samples = x
        return samples


    def get_loss(self, x, code):
        loss = self.diffusion.get_loss(x, code)
        return loss
