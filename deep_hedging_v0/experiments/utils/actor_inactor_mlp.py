import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hedging.logit_normal import LogitNormal
from experiments.utils.ppo_mlp_actor import create_ppo_mlp_actor

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule,TensorDictSequential
from torchrl.modules import ProbabilisticActor
import torch
import torch.nn as nn
from torchrl.modules import (
    SafeModule, 
    NormalParamExtractor, 
    ProbabilisticActor, 
    TanhNormal,
    ValueOperator,
    ActorValueOperator
)
from torchrl.data import Bounded
from tensordict.nn import TensorDictModule

def generate_actor_inactor_mlp(feat_dim, hidden_dim, action_dim, env, device,
                               action_low=0.0, action_high=1.0):
    
    actor_model = create_ppo_mlp_actor(input_dim=feat_dim,
                                        action_dim=action_dim,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        action_high=action_high,
                                        action_low=action_low)

    class FeatureExtractor(nn.Module):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.mlp(x)

    inactor_feature_extractor = SafeModule(
        module=FeatureExtractor(),
        in_keys=["observation"],
        out_keys=["i_feature"],
    )

    inactor_policy_network = TensorDictModule(
        nn.Sequential(torch.nn.Linear(hidden_dim, 1)),
        in_keys=["i_feature"],
        out_keys=["logits"],
    )

    inactor = ProbabilisticActor(
        module=inactor_policy_network,
        in_keys=["logits"],
        out_keys=["inact"],
        distribution_class=torch.distributions.Bernoulli,
        return_log_prob=True,
    )

    inactor_critic = ValueOperator(
        module=nn.Sequential(torch.nn.Linear(hidden_dim, int(hidden_dim/2)), 
                             nn.Tanh(), 
                             nn.Linear(int(hidden_dim/2), 1),
                             nn.Flatten(start_dim=-2)
                             ),
        in_keys=["i_feature"],
        out_keys=["i_state_value"],
    )

    inactor_model = ActorValueOperator(inactor_feature_extractor, inactor, inactor_critic).to(device)

    return actor_model, inactor_model

