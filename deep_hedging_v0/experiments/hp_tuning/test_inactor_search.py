"""Regression checks for the isolated BSM call MLP search."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchrl.envs import GymWrapper
from torchrl.modules import IndependentNormal

from experiments.utils.inactor_search import (
    PPOUpdater,
    SearchJointPolicy,
    collect_episode,
    make_search_actor,
    make_search_inactor,
)
from experiments.utils.sim_config import PPOConfig
from hedging.envs import HedgeCallBS


class ResumeGridChangeTests(unittest.TestCase):
    """A resumed search may change the trial grid but never the core settings."""

    def make_args(self, output, resume=False, inactor_lrs=(1e-3,), num_paths=2):
        return argparse.Namespace(
            output=output, resume=resume, dry_run=False, smoke=False, seeds=[0],
            inactor_lrs=list(inactor_lrs), strategies=["reference"],
            warmup_epochs=1, episodes_per_epoch=1, joint_batches=2, rungs=[1, 2],
            reduction=1, finalists=1, num_paths=num_paths, eval_paths=2,
            validation_seeds=[100001], test_seeds=[200003], sub_batches=2,
            eval_every=1, checkpoint_every=1, log_every=1, threads=1, device="cpu",
        )

    def test_grid_change_resume_reuses_legacy_warmup(self):
        from experiments.hp_tuning.search_inactive_bsm_hedge_call_mlp import InactorSearch, Trial

        with tempfile.TemporaryDirectory(prefix="bsm_inactor_grid_change_") as temporary:
            output = Path(temporary) / "run"
            trial = Trial("reference", inactor_lr=1e-3)
            search = InactorSearch(self.make_args(output), [trial])
            search.warmup(0)
            search.train_trial(trial, 0, 1)

            # Simulate a run created before the flexible-grid change: its manifest
            # and checkpoints carry a signature the new code cannot recompute.
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["signature"] = "legacy-signature"
            manifest_path.write_text(json.dumps(manifest))
            for checkpoint in output.rglob("*.pt"):
                state = torch.load(checkpoint, weights_only=True, map_location="cpu")
                state["signature"] = "legacy-signature"
                torch.save(state, checkpoint)

            resumed = InactorSearch(
                self.make_args(output, resume=True, inactor_lrs=(1e-3, 3e-3)),
                [trial, Trial("reference", inactor_lr=3e-3)],
            )
            self.assertIn("legacy-signature", resumed.signature_history)
            resumed.warmup(0)  # accepted via signature history; nothing retrained
            cached = resumed.train_trial(trial, 0, 1)
            self.assertEqual(cached["budget"], 1)
            fresh = resumed.train_trial(Trial("reference", inactor_lr=3e-3), 0, 1)
            self.assertEqual(fresh["budget"], 1)
            updated = json.loads(manifest_path.read_text())
            self.assertEqual(updated["signature"], resumed.signature)
            self.assertIn("legacy-signature", updated["signature_history"])

    def test_core_setting_change_resume_is_refused(self):
        from experiments.hp_tuning.search_inactive_bsm_hedge_call_mlp import InactorSearch, Trial

        with tempfile.TemporaryDirectory(prefix="bsm_inactor_core_change_") as temporary:
            output = Path(temporary) / "run"
            InactorSearch(self.make_args(output), [Trial("reference")])
            with self.assertRaisesRegex(ValueError, "num_paths"):
                InactorSearch(
                    self.make_args(output, resume=True, num_paths=3), [Trial("reference")]
                )


class InactorSearchTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"DEEP_HEDGING_PATH_CACHE": "0"})
        environment.start()
        self.addCleanup(environment.stop)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.env = GymWrapper(HedgeCallBS(
            np.array([100.]), np.array([[100.]]), 1., .01, np.array([.2]),
            4, 4, history_len=1, transaction_cost=True, transaction_fee_rate=.001,
        ), device="cpu")
        self.addCleanup(self.env.close)
        self.actor = make_search_actor(11, 16, "cpu")
        self.policy = SearchJointPolicy(self.actor.get_policy_operator())

    def collect(self, policy=None, seed=123):
        return collect_episode(self.env, policy or self.policy, 4, seed, seed + 500_000)

    def saturate_actor(self):
        head = next(module for module in self.actor.module[1].modules()
                    if isinstance(module, torch.nn.Linear))
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
            head.bias[0] = 20.0

    def test_observations_and_next_observations_are_snapshots(self):
        initial = self.env.reset(seed=123)["observation"].clone()
        batch = self.collect()
        self.assertTrue(torch.equal(initial, batch["observation"][:, 0]))
        self.assertFalse(torch.equal(batch["observation"][:, 0], batch["observation"][:, -1]))
        self.assertTrue(torch.equal(
            batch["next", "observation"][:, :-1], batch["observation"][:, 1:]
        ))
        saved = batch.clone()
        self.collect(seed=456)
        self.assertTrue(torch.equal(saved["observation"], batch["observation"]))
        self.assertTrue(torch.equal(saved["next", "observation"], batch["next", "observation"]))

    def test_unchanged_policy_reproduces_recorded_likelihoods(self):
        for saturated in (False, True):
            with self.subTest(saturated=saturated):
                if saturated:
                    self.saturate_actor()
                batch = self.collect()
                with torch.no_grad():
                    distribution = self.actor.get_policy_operator().get_dist(batch.clone())
                    log_ratio = distribution.log_prob(batch["latent_action"]) - batch["latent_action_log_prob"]
                torch.testing.assert_close(log_ratio, torch.zeros_like(log_ratio), atol=1e-5, rtol=0)
                torch.testing.assert_close(batch["action"], batch["latent_action"].tanh())

    def test_saturation_does_not_destroy_stored_gaussian_samples(self):
        self.saturate_actor()
        batch = self.collect()
        self.assertEqual(batch["action"].unique().numel(), 1)
        self.assertGreater(batch["latent_action"].unique().numel(), 1)
        self.assertTrue(batch["latent_action_log_prob"].isfinite().all())

    def test_squashed_entropy_stays_finite_with_correct_gradient(self):
        updater = PPOUpdater(self.actor, PPOConfig(), 1e-4)
        location = torch.tensor([[20.], [-20.]], requires_grad=True)
        distribution = IndependentNormal(location, torch.full_like(location, .5))
        entropy = updater.loss._get_entropy(distribution, torch.Size([2]))
        self.assertTrue(entropy.isfinite().all())
        self.assertTrue((entropy < -30).all())
        entropy.sum().backward()
        torch.testing.assert_close(location.grad, torch.tensor([[-2.], [2.]]))

    def test_gate_off_matches_actor_only_with_paired_randomness(self):
        gate = make_search_inactor(11, 16, "cpu", .05, contextual=True)
        policy = SearchJointPolicy(
            self.actor.get_policy_operator(), gate.get_policy_operator(), gate_strength=0.0
        )
        control = self.collect()
        gated = self.collect(policy)
        self.assertTrue(torch.equal(control["action"], gated["action"]))
        self.assertTrue(torch.equal(control["next", "reward"], gated["next", "reward"]))

    def test_saturated_actor_and_inactor_updates_keep_finite_gradients(self):
        self.saturate_actor()
        gate = make_search_inactor(11, 16, "cpu", .05, contextual=True)
        policy = SearchJointPolicy(self.actor.get_policy_operator(), gate.get_policy_operator())
        actor_updater = PPOUpdater(self.actor, PPOConfig(), 1e-4)
        gate_updater = PPOUpdater(gate, PPOConfig(), 1e-3, inactor=True)
        for seed in range(123, 126):
            batch = self.collect(policy, seed)
            executed = batch["action"].clone()
            for updater in (gate_updater, actor_updater):
                metrics = updater.update(batch, 2)
                self.assertTrue(math.isfinite(metrics["grad_norm"]))
                self.assertTrue(math.isfinite(metrics["loss_total"]))
                self.assertEqual(metrics["optimizer_steps"], 2)
            self.assertTrue(torch.equal(executed, batch["action"]))


if __name__ == "__main__":
    unittest.main()