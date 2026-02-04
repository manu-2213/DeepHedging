import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import (
    ProbabilisticActor, 
    ValueOperator, 
    ActorValueOperator,
    SafeModule,
    TanhNormal)
from hedging.logit_normal import LogitNormal


def create_ppo_rnn_actor(input_dim, action_dim, hidden_dim=64, num_layers=2):
    class FeatureExtractor(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super(FeatureExtractor, self).__init__()
            self.rnn = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0
            )

        def forward(self, x):
            if len(x.shape) > 3:  # Handle 4D input
                x_reshaped = x.view(-1, x.shape[-2], x.shape[-1])
                output, _ = self.rnn(x_reshaped)
                # Reshape output back to original batch dimensions
                output = output.view(
                    x.shape[0], x.shape[1], x.shape[2], output.shape[-1]
                )
            else:  # Handle 3D input directly
                output, _ = self.rnn(x)
            output = output[..., -1, :]  # Take output from the last time step
            return output
        
    feature_extractor = TensorDictModule(
        module=FeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers),
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

    actor = ProbabilisticActor(
        module=policy_network,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=LogitNormal,
        return_log_prob=True,
    )

    critic = ValueOperator(
        module=nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim/2), 1)
        ),
        in_keys=["feature"],
        out_keys=["state_value"],
    )

    model = ActorValueOperator(feature_extractor, actor, critic)

    return model

def create_ppo_rnn_actor_exotic(input_dim, action_dim, hidden_dim):

    class FeatureExtractor(nn.Module):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            self.rnn = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.0
            )

        def forward(self, x):
            if len(x.shape) > 3:  # Handle 4D input
                x_reshaped = x.view(-1, x.shape[-2], x.shape[-1])
                output, _ = self.rnn(x_reshaped)
                # Reshape output back to original batch dimensions
                output = output.view(
                    x.shape[0], x.shape[1], x.shape[2], output.shape[-1]
                )
            else:  # Handle 3D input directly
                output, _ = self.rnn(x)
            output = output[..., -1, :]  # Take output from the last time step
            return output

    feature_extractor = SafeModule(
        module=FeatureExtractor(),
        in_keys=["observation"],
        out_keys=["feature"],
    )
    policy_network = TensorDictModule(
        nn.Sequential(torch.nn.Linear(hidden_dim, 2*action_dim), NormalParamExtractor()),
        in_keys=["feature"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=policy_network,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    critic = ValueOperator(
        module=nn.Sequential(torch.nn.Linear(hidden_dim, 8), nn.Tanh(), nn.Linear(8, 1)),
        in_keys=["feature"],
        out_keys=["state_value"],
    )
    model = ActorValueOperator(feature_extractor, actor, critic)

    return model

