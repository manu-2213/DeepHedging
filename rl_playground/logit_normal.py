import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform


class LogitNormal(TransformedDistribution):

    def __init__(self, loc, scale):
        loc = loc.squeeze(-1)
        scale = scale.squeeze(-1)
        self.loc = loc
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, SigmoidTransform())

    @property
    def mode(self):
        return torch.sigmoid(self.loc)

    @property
    def deterministic_sample(self):
        return torch.sigmoid(self.loc)
