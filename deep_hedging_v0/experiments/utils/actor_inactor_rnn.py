import torch
import torch.nn as nn
from torchrl.modules import (
    SafeModule, 
    NormalParamExtractor, 
    ProbabilisticActor, 
    TanhNormal,
    ValueOperator,
    ActorValueOperator,
    TanhNormal
)
from torchrl.data import Bounded
from tensordict.nn import TensorDictModule, TensorDictSequential, NormalParamExtractor

# Handle multiple action_dim for bernoulli

class MultiBernoulli(torch.distributions.Independent):
    def __init__(self, logits):
        base = torch.distributions.Bernoulli(logits=logits)
        super().__init__(base, 1) # reinterprest action_dim

def generate_actor_inactor_rnn(feat_dim, hidden_dim, action_dim, env, device, action_low = 0.0,
                               action_high = 1.0):

    class FeatureExtractor(nn.Module):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            self.rnn = nn.LSTM(
                input_size=feat_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.0
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

    actor_feature_extractor = SafeModule(
        module=FeatureExtractor(),
        in_keys=["observation"],
        out_keys=["a_feature"],
    )

    inactor_feature_extractor = SafeModule(
        module=FeatureExtractor(),
        in_keys=["observation"],
        out_keys=["i_feature"],
    )

    actor_policy_network = TensorDictModule(
        nn.Sequential(torch.nn.Linear(hidden_dim, 2*action_dim), NormalParamExtractor()),
        in_keys=["a_feature"],
        out_keys=["loc", "scale"],
    )

    inactor_policy_network = TensorDictModule(
        nn.Sequential(torch.nn.Linear(hidden_dim, action_dim)),
        in_keys=["i_feature"],
        out_keys=["logits"],
    )

    action_spec = Bounded(
        low=action_low,
        high=action_high,
        shape=(action_dim,),
        dtype=torch.float,
        device=device,
    )

    actor = ProbabilisticActor(
        module=actor_policy_network,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        spec=action_spec,
    )

    inactor = ProbabilisticActor(
        module=inactor_policy_network,
        in_keys=["logits"],
        out_keys=["inact"],
        distribution_class=torch.distributions.Bernoulli,
        return_log_prob=True,
    )

    actor_critic = ValueOperator(
        module=nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim/2)), 
                             nn.Tanh(), 
                             nn.Linear(int(hidden_dim/2), 1)),
        in_keys=["a_feature"],
        out_keys=["a_state_value"],
    )

    inactor_critic = ValueOperator(
        module=nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim/2)), 
                             nn.Tanh(), 
                             nn.Linear(int(hidden_dim/2), 1)),
        in_keys=["i_feature"],
        out_keys=["i_state_value"],
    )


    actor_model = ActorValueOperator(actor_feature_extractor, actor, actor_critic).to(device)
    inactor_model = ActorValueOperator(inactor_feature_extractor, inactor, inactor_critic).to(device)

    return actor_model, inactor_model


def generate_actor_inactor_rnn_exotic(feat_dim, hidden_dim, action_dim, env, device):
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            self.rnn = nn.LSTM(
                input_size=feat_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.0
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

    actor_feature_extractor = SafeModule(
        module=FeatureExtractor(),
        in_keys=["observation"],
        out_keys=["a_feature"],
    )
    inactor_feature_extractor = SafeModule(
        module=FeatureExtractor(),
        in_keys=["observation"],
        out_keys=["i_feature"]
    )
    actor_policy_network = TensorDictModule(
        nn.Sequential(torch.nn.Linear(hidden_dim, 2*action_dim), NormalParamExtractor()),
        in_keys=["a_feature"],
        out_keys=["loc", "scale"],
    )
    inactor_policy_network = TensorDictModule(
        nn.Sequential(torch.nn.Linear(hidden_dim, action_dim)),
        in_keys=["i_feature"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=actor_policy_network,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal, # No need to bound action spec for exotics [-1,1] range
        return_log_prob=True,
    )
    inactor = ProbabilisticActor(
        module=inactor_policy_network,
        in_keys=["logits"],
        out_keys=["inact"],
        distribution_class=MultiBernoulli,
        return_log_prob=True,
    )
    actor_critic = ValueOperator(
        module=nn.Sequential(nn.Linear(hidden_dim, 8), nn.Tanh(), nn.Linear(8, 1)),
        in_keys=["a_feature"],
        out_keys=["a_state_value"],
    )
    inactor_critic = ValueOperator(
        module=nn.Sequential(nn.Linear(hidden_dim, 8), nn.Tanh(), nn.Linear(8, 1)),
        in_keys=["i_feature"],
        out_keys=["i_state_value"],
    )
    actor_model = ActorValueOperator(actor_feature_extractor, actor, actor_critic)
    inactor_model = ActorValueOperator(inactor_feature_extractor, inactor, inactor_critic)

    return actor_model, inactor_model


    