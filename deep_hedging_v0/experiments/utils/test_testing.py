import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from experiments.utils import testing
from experiments.utils.actor_inactor_mlp import generate_actor_inactor_mlp
from experiments.utils.actor_inactor_rnn import generate_actor_inactor_rnn_exotic
from experiments.utils.joint_policy import JointPolicy
from hedging.envs import HedgeCallBS


class ConstantOutput(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, observation):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Evaluation must disable training and gradients")
        return torch.full_like(observation[..., :1], self.value)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {
            "DEEP_HEDGING_SAVE_ARTIFACTS": "0",
            "DEEP_HEDGING_PATH_CACHE": "0",
        }))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.log = self.enterContext(patch.object(testing.wandb, "log"))
        self.device = torch.device("cpu")

    def make_env(self, history_len=1):
        return HedgeCallBS(
            np.array([100.0]), np.array([[100.0]]), maturity=0.25,
            r=0.01, sigma=np.array([0.2]), num_paths=16, num_steps=5,
            history_len=history_len, transaction_cost=True,
        )

    def make_policy(self, probability=0.4):
        actor = TensorDictModule(
            ConstantOutput(0.5), in_keys=["observation"], out_keys=["action"],
        )
        gate = ProbabilisticActor(
            TensorDictModule(
                ConstantOutput(probability),
                in_keys=["observation"], out_keys=["probs"],
            ),
            in_keys=["probs"], out_keys=["inact"],
            distribution_class=torch.distributions.Bernoulli,
        )
        return JointPolicy(actor, gate, 1, self.device)

    def test_final_metrics_and_plots_match_primary_rollout(self):
        base_env = self.make_env()
        policy = self.make_policy()
        paths = []
        masks = []
        rollouts = []
        original = testing._evaluation_rollout

        def capture(env, eval_policy, num_steps):
            rollout = original(env, eval_policy, num_steps)
            paths.append(base_env.stock_prices.copy())
            masks.append(rollout["inact"].clone())
            rollouts.append(rollout)
            return rollout

        with patch.object(testing, "_evaluation_rollout", side_effect=capture), \
                patch.object(testing, "plot_portfolio_vs_option_price"), \
                patch.object(testing, "_log_pnl_distribution"), \
                patch.object(testing, "_log_delta_paths"), \
                patch.object(testing, "_log_transaction_cost_scatter") as scatter, \
                patch.object(testing, "_log_action_magnitude_distribution") as actions:
            pnl, metrics = testing.test_model(
                base_env, policy, 5, self.device, plotting=True,
            )

        payload = {
            key: value for call in self.log.call_args_list
            for key, value in call.args[0].items()
        }
        for name, value in metrics.items():
            np.testing.assert_equal(payload[f"final/{name}"], value)
            np.testing.assert_equal(payload[f"final/inaction_stochastic/{name}"], value)
            self.assertIn(f"final/inaction_deterministic/{name}", payload)
        np.testing.assert_array_equal(paths[0], paths[1])
        self.assertEqual(masks[0].unique().tolist(), [0.0])
        self.assertEqual(masks[1].unique().tolist(), [0.0, 1.0])
        np.testing.assert_array_equal(
            pnl, (base_env.portfolio_value[..., -1] - base_env.option_prices[..., -1]).flatten(),
        )
        np.testing.assert_array_equal(scatter.call_args.args[1], pnl)
        self.assertIs(actions.call_args.args[0], rollouts[-1])

    def test_repeatability_and_state_restoration(self):
        policy = self.make_policy()
        rng_state = torch.random.get_rng_state().clone()
        first, _ = testing.test_model(self.make_env(), policy, 5, self.device)
        self.assertTrue(torch.equal(rng_state, torch.random.get_rng_state()))
        self.assertTrue(all(module.training for module in policy.modules()))
        policy.prev_action = torch.full((16, 1), 99.0)
        with set_exploration_type(ExplorationType.RANDOM):
            second, _ = testing.test_model(self.make_env(), policy, 5, self.device)
        np.testing.assert_array_equal(first, second)

    def test_always_trade_matches_ordinary_actor(self):
        policy = self.make_policy(probability=0.0)
        joint_pnl, joint_metrics = testing.test_model(self.make_env(), policy, 5, self.device)
        plain_pnl, plain_metrics = testing.test_model(
            self.make_env(), policy.actor_policy, 5, self.device,
        )
        np.testing.assert_array_equal(joint_pnl, plain_pnl)
        for name, value in plain_metrics.items():
            np.testing.assert_equal(joint_metrics[name], value)

    def test_rollout_resets_once_and_holds_previous_action(self):
        env = GymWrapper(self.make_env(), device=self.device)
        policy = self.make_policy(probability=1.0)
        policy.prev_action = torch.full((16, 1), 99.0)
        with patch.object(env, "reset", wraps=env.reset) as reset:
            rollout = testing._evaluation_rollout(env, policy, 5)
        reset.assert_called_once_with(seed=0)
        self.assertTrue(torch.all(rollout["action"] == 0))
        self.assertTrue(torch.all(rollout["original_action"] == 0.5))
        self.assertEqual(env._env.current_step, 5)

    def test_mlp_and_rnn_models(self):
        for factory, history_len in (
            (generate_actor_inactor_mlp, 1),
            (generate_actor_inactor_rnn_exotic, 3),
        ):
            with self.subTest(factory=factory.__name__):
                base_env = self.make_env(history_len)
                env = GymWrapper(base_env, device=self.device)
                actor, inactor = factory(
                    feat_dim=env.observation_spec["observation"].shape[-1],
                    hidden_dim=8, action_dim=1, env=env, device=self.device,
                )
                policy = JointPolicy(
                    actor.get_policy_operator(), inactor.get_policy_operator(),
                    1, self.device,
                )
                pnl, metrics = testing.test_model(base_env, policy, 5, self.device)
                self.assertEqual(pnl.size, 16)
                self.assertTrue(np.isfinite(metrics["mean"]))


if __name__ == "__main__":
    unittest.main()