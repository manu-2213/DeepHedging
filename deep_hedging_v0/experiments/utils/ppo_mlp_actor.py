import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule, NormalParamExtractor, TensorDictSequential
from torchrl.modules import ProbabilisticActor, ValueOperator, ActorValueOperator, TanhNormal, SafeProbabilisticModule, SafeModule
from torchrl.data import Bounded
from hedging.logit_normal import LogitNormal


def create_ppo_mlp_actor(input_dim, action_dim, hidden_dim, device, action_high=1.0, action_low=0.0):
    class FeatureExtractor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            if x.ndim >= 3 and x.shape[-2] == 1:
                x = x.squeeze(-2)   # (600, 250, 11) instead of (600, 250, 1, 11)
            return self.net(x)  


    feature_extractor = TensorDictModule(
        module=FeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim),
        in_keys=["observation"],
        out_keys=["feature"],
    )

    policy_network = TensorDictModule(
        nn.Sequential(
            nn.Linear(hidden_dim, 2 * action_dim),
            NormalParamExtractor(),  
        ),
        in_keys=["feature"],
        out_keys=["loc", "scale"],
    )

    action_spec = Bounded(
        low=action_low,
        high=action_high,
        shape=(action_dim,),
        dtype=torch.float,
        device=device,
    )

    actor = ProbabilisticActor(
        module=policy_network,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        spec=action_spec,
    )

    critic = ValueOperator(
        module=nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        ),
        in_keys=["feature"],
        out_keys=["state_value"],
    )

    model = ActorValueOperator(feature_extractor, actor, critic)

    return model