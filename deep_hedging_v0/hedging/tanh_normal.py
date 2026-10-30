import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform



class TanhNorm(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        # Squeeze to match LogitNormal's dimensionality
        loc = loc.squeeze(-1)
        scale = scale.squeeze(-1)
        self.loc = loc
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, TanhTransform(cache_size=1), validate_args=validate_args)

    @property
    def mode(self):
        return torch.tanh(self.loc)

    @property
    def deterministic_sample(self):
        return self.mode