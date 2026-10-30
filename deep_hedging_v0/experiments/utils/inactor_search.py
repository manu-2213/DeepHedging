"""Isolated policy components for the BSM call MLP inactor search."""
from __future__ import annotations

import math

import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictParams
from torch import nn
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ActorValueOperator, IndependentNormal, ProbabilisticActor, SafeModule, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from experiments.utils.ppo_mlp_actor import create_ppo_mlp_actor


def make_search_actor(feat_dim, hidden_dim, device):
    """Reuse the actor networks, retaining Gaussian samples before tanh rounds them."""
    actor = create_ppo_mlp_actor(feat_dim, 1, hidden_dim, device)
    policy = ProbabilisticActor(
        module=actor.module[1].module[0],
        in_keys=["loc", "scale"],
        out_keys=["latent_action"],
        distribution_class=IndependentNormal,
        return_log_prob=True,
    )
    return ActorValueOperator(actor.module[0], policy, actor.module[2]).to(device)


class SearchActorPPOLoss(ClipPPOLoss):
    """Gaussian likelihood ratios with entropy of the executed tanh actions."""

    actor_network: TensorDictModuleBase
    critic_network: TensorDictModuleBase
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def _get_entropy(self, dist, adv_shape):
        latent = dist.rsample((self.samples_mc_entropy,))
        log_jacobian = torch.distributions.transforms.TanhTransform().log_abs_det_jacobian(
            latent, latent.tanh()
        )
        entropy = dist.entropy() + log_jacobian.sum(dim=-1).mean(dim=0)
        return entropy.unsqueeze(-1)


class InactorFeatures(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observation):
        if observation.ndim >= 3 and observation.shape[-2] == 1:
            observation = observation.squeeze(-2)
        return self.mlp(observation)


class InactorProbability(nn.Module):
    """Include the activation schedule in both sampling and PPO likelihoods."""

    def __init__(self, hidden_dim, initial_hold_probability=None, contextual=False):
        super().__init__()
        self.contextual = contextual
        self.linear = nn.Linear(hidden_dim + (2 if contextual else 0), 1)
        if initial_hold_probability is not None:
            if not 0 < initial_hold_probability < 1:
                raise ValueError("initial_hold_probability must be strictly between 0 and 1")
            nn.init.zeros_(self.linear.weight)
            nn.init.constant_(
                self.linear.bias,
                math.log(initial_hold_probability / (1 - initial_hold_probability)),
            )

    def forward(self, features, gate_strength, original_action=None, previous_action=None):
        if self.contextual:
            features = torch.cat(
                (features, original_action, original_action - previous_action), dim=-1
            )
        return self.linear(features).sigmoid() * gate_strength


def make_search_inactor(
    feat_dim, hidden_dim, device, initial_hold_probability=None, contextual=False
):
    feature_extractor = SafeModule(
        InactorFeatures(feat_dim, hidden_dim),
        in_keys=["observation"],
        out_keys=["i_feature"],
    )
    head_keys = ["i_feature", "gate_strength"]
    if contextual:
        head_keys += ["original_action", "previous_action"]
    policy = ProbabilisticActor(
        module=TensorDictModule(
            InactorProbability(hidden_dim, initial_hold_probability, contextual),
            in_keys=head_keys,
            out_keys=["hold_probability"],
        ),
        in_keys={"probs": "hold_probability"},
        out_keys=["inact"],
        distribution_class=torch.distributions.Bernoulli,
        return_log_prob=True,
    )
    critic = ValueOperator(
        module=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        ),
        in_keys=["i_feature"],
        out_keys=["i_state_value"],
    )
    return ActorValueOperator(feature_extractor, policy, critic).to(device)


class SearchJointPolicy(TensorDictModuleBase):
    """Retain actor proposals and reset holdings independently of PPO buffers."""

    def __init__(self, actor_policy, inactor_policy=None, gate_strength=1.0):
        super().__init__()
        self.in_keys = ["observation"]
        self.out_keys = [
            "action", "original_action", "previous_action", "gate_strength",
            "inact", "hold_probability", "latent_action", "latent_action_log_prob",
        ]
        if inactor_policy is not None:
            self.out_keys.append("inact_log_prob")
        self.actor_policy = actor_policy
        self.inactor_policy = inactor_policy
        self.gate_strength = gate_strength
        self.prev_action = None

    def reset(self):
        self.prev_action = None

    def forward(self, data):
        data = data.clone(False)
        data.set("observation", data["observation"].clone())
        data = self.actor_policy(data)
        proposal = data["latent_action"].tanh()
        previous = self.prev_action
        if previous is None or previous.shape != proposal.shape:
            previous = torch.zeros_like(proposal)
        is_init = data.get("is_init", None)
        if is_init is not None:
            previous = torch.where(is_init.reshape_as(proposal), 0.0, previous)
        data.set("original_action", proposal.clone())
        data.set("previous_action", previous.clone())
        data.set("gate_strength", torch.full_like(proposal, self.gate_strength))
        if self.inactor_policy is None:
            data.set("inact", torch.bernoulli(torch.zeros_like(proposal)))
            data.set("hold_probability", torch.zeros_like(proposal))
        else:
            data = self.inactor_policy(data)
        final_action = torch.where(data["inact"].reshape_as(proposal) > 0.5, previous, proposal)
        data.set("action", final_action)
        self.prev_action = final_action.detach().clone()
        return data


class PPOUpdater:
    """One on-policy update, without copying full rollouts into replay buffers."""

    def __init__(
        self, model, config, learning_rate, *, inactor=False,
        entropy_coeff=None, clip_epsilon=None, normalize_advantage=False, target_kl=0.0,
    ):
        self.inactor = inactor
        self.target_kl = target_kl
        value_key = "i_state_value" if inactor else "state_value"
        self.advantage = GAE(
            gamma=config.gamma, lmbda=config.lmbda,
            value_network=model.get_value_operator(), shifted=True,
        )
        self.advantage.set_keys(value=value_key)
        loss_class = ClipPPOLoss if inactor else SearchActorPPOLoss
        self.loss = loss_class(
            actor_network=model.get_policy_operator(),
            critic_network=model.get_value_operator(),
            clip_epsilon=config.clip_param if clip_epsilon is None else clip_epsilon,
            entropy_coeff=config.entropy_coeff if entropy_coeff is None else entropy_coeff,
            critic_coeff=config.value_coef,
            normalize_advantage=normalize_advantage,
        )
        self.loss.set_keys(
            value=value_key,
            action="inact" if inactor else "latent_action",
            sample_log_prob="inact_log_prob" if inactor else "latent_action_log_prob",
        )
        self.optimizer = torch.optim.Adam(self.loss.parameters(), lr=learning_rate)

    def update(self, batch, sub_batches, passes=1, critic_only=False):
        data = batch.clone(False)
        with torch.no_grad():
            self.advantage(data)
        flattened = data.reshape(-1)
        totals = {}
        updates = 0
        stopped = False
        for _ in range(passes):
            permutation = torch.randperm(flattened.numel(), device=flattened.device)
            for indices in permutation.tensor_split(min(sub_batches, flattened.numel())):
                losses = self.loss(flattened[indices])
                total = losses["loss_critic"] if critic_only else sum(
                    value for key, value in losses.items() if key.startswith("loss_")
                )
                if not torch.isfinite(total).all():
                    raise FloatingPointError("Non-finite PPO loss")
                kl = float(losses.get("kl_approx", torch.zeros(())).detach().mean())
                if not math.isfinite(kl):
                    raise FloatingPointError("Non-finite PPO KL estimate")
                if not critic_only and self.target_kl and kl > self.target_kl:
                    stopped = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss.parameters(), 1.0, error_if_nonfinite=True
                )
                self.optimizer.step()
                updates += 1
                measurements = {
                    "loss_total": float(total.detach()), "kl_approx": kl,
                    "grad_norm": float(gradient_norm),
                }
                measurements.update({
                    key: float(value.detach().mean())
                    for key, value in losses.items()
                    if key.startswith("loss_") or key == "clip_fraction"
                })
                for key, value in measurements.items():
                    totals[key] = totals.get(key, 0.0) + value
            if stopped:
                break
        return {
            **{key: value / max(updates, 1) for key, value in totals.items()},
            "optimizer_steps": updates, "kl_stopped": stopped,
        }


def collect_episode(env, policy, num_steps, path_seed, policy_seed, deterministic=False):
    policy.reset()
    torch.manual_seed(policy_seed)
    initial = env.reset(seed=path_seed)
    exploration = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    with torch.no_grad(), set_exploration_type(exploration):
        batch = env.rollout(
            num_steps, policy=policy, auto_reset=False, tensordict=initial,
            break_when_any_done=False, auto_cast_to_device=True,
            callback=lambda wrapped_env, transition: transition.set("next", transition["next"].clone()),
        )
    if batch.batch_size[-1] != num_steps:
        raise RuntimeError("Expected one complete episode per rollout")
    return batch


def rollout_metrics(batch):
    rewards = batch["next", "reward"].squeeze(-1)
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Non-finite environment rewards")
    tracking = -rewards.mean(dim=-1).detach().cpu().numpy()
    var95 = float(np.percentile(tracking, 95))
    tail = tracking[tracking >= var95]
    action = batch["action"]
    return {
        "avg_reward": float(rewards.mean()),
        "tracking_var95": var95,
        "tracking_cvar95": float(tail.mean()),
        "terminal_abs_error": float(-rewards[..., -1].mean()),
        "hold_rate": float(batch["inact"].float().mean()),
        "hold_probability": float(batch["hold_probability"].mean()),
        "actor_saturation_rate": float((batch["original_action"].abs() > 0.9999).float().mean()),
        "actor_loc_abs_max": float(batch["loc"].abs().max()),
        "actor_scale_min": float(batch["scale"].min()),
        "actor_scale_max": float(batch["scale"].max()),
        "intervention_rate": float(
            ((action - batch["original_action"]).abs() > 1e-6).float().mean()
        ),
        "turnover": float((action - batch["previous_action"]).abs().mean()),
    }


def evaluate_policy(env, policy, num_steps, seeds, deterministic=False):
    """Use common paths and policy seeds; average risk metrics across replicates."""
    records = []
    was_training = policy.training
    policy.eval()
    try:
        for seed in seeds:
            batch = collect_episode(
                env, policy, num_steps, seed, seed + 500_000, deterministic
            )
            records.append({"path_seed": seed, **rollout_metrics(batch)})
    finally:
        policy.train(was_training)
        policy.reset()
    return {
        **{
            key: float(np.mean([record[key] for record in records]))
            for key in records[0] if key != "path_seed"
        },
        "reward_std_across_path_seeds": float(np.std(
            [record["avg_reward"] for record in records], ddof=1 if len(records) > 1 else 0
        )),
        "replicates": records,
    }