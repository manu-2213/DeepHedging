import os
import sys
import numpy as np
import torch
import wandb
import argparse

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hedging.envs import HedgeDocHeston
from experiments.utils.ppo_rnn_actor import create_ppo_rnn_actor_exotic
from experiments.utils.training_loop import action_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    DEFAULT_WANDB_PROJECT,
    PPOConfig,
    TrainingConfig,
    get_heston_doc_config,
    training_to_wandb_config,
)

from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = get_heston_doc_config()
    cfg["history_len"] = 5
    training_cfg = TrainingConfig(num_epochs=20, num_episodes=200)
    ppo_cfg = PPOConfig(entropy_coeff=0.0, learning_rate=1e-4)

    def build_env(num_paths: int) -> HedgeDocHeston:
        return HedgeDocHeston(
            S0=cfg["S0"],
            K=cfg["K"],
            H=cfg["H"],
            r=cfg["r"],
            v0=cfg["v0"],
            theta=cfg["params"]["theta"],
            rho=cfg["params"]["rho"],
            kappa=cfg["params"]["kappa"],
            xi=cfg["params"]["sigma"],
            maturity=cfg["maturity"],
            num_steps=cfg["num_steps"],
            num_paths=num_paths,
            history_len=cfg["history_len"],
            transaction_cost=cfg["transaction_cost"],
            transaction_fee_rate=cfg["transaction_fee_rate"],
        )

    env = GymWrapper(build_env(cfg["num_paths"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_per_batch = env.num_envs * cfg["num_steps"]
    sub_batch_num = training_cfg.sub_batch_num
    sub_batch_size = frames_per_batch // sub_batch_num

    model = create_ppo_rnn_actor_exotic(
        input_dim=cfg["input_dim"],
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_size"],
    ).to(device)

    advantage_module = GAE(
        gamma=ppo_cfg.gamma,
        lmbda=ppo_cfg.lmbda,
        value_network=model.get_value_operator(),
        shifted=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=model.get_policy_operator(),
        critic_network=model.get_value_operator(),
        clip_epsilon=ppo_cfg.clip_param,
        entropy_coeff=ppo_cfg.entropy_coeff,
        value_coef=ppo_cfg.value_coef,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=ppo_cfg.learning_rate)

    problem_name = "heston_doc_rnn"
    wandb_config = training_to_wandb_config(
        training_cfg,
        ppo_cfg,
        extra={
            "problem": problem_name,
            "seed": seed,
            "frames_per_batch": frames_per_batch,
            "sub_batch_size": sub_batch_size,
            "num_paths": cfg["num_paths"],
            "num_steps": cfg["num_steps"],
            "history_len": cfg["history_len"],
            "input_dim": cfg["input_dim"],
            "action_dim": cfg["action_dim"],
            "transaction_cost": cfg["transaction_cost"],
            "transaction_fee_rate": cfg["transaction_fee_rate"],
        },
    )

    wandb.init(
        project=DEFAULT_WANDB_PROJECT,
        name=f"{problem_name}_seed{seed}",
        group=problem_name,
        config=wandb_config,
    )

    model = action_training(
        env,
        model,
        training_cfg.num_epochs,
        training_cfg.num_episodes,
        device,
        advantage_module,
        loss_module,
        optim,
        frames_per_batch,
        sub_batch_num,
        sub_batch_size,
        log_frquency=1,
    )

    test_env = build_env(num_paths=5)
    test_model(test_env, model, cfg["num_steps"], device, plotting=False)

    wandb.finish()


if __name__ == "__main__":
    main()
