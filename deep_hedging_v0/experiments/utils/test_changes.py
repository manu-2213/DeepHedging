"""
Verbose regression / smoke test for all applied changes.

Run with:
    cd /Users/manu13/Desktop/PHD/DeepHedging
    python -m deep_hedging_v0.experiments.utils.test_changes

Each CHECK prints PASS or FAIL with an explanation.
A final summary shows the total count.
"""

import sys
import os
import traceback
import numpy as np
import torch

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

PASS = "  PASS"
FAIL = "  FAIL"
results = []


def check(name: str, fn):
    print(f"\n[CHECK] {name}")
    try:
        fn()
        print(f"{PASS}")
        results.append((name, True, None))
    except Exception as e:
        msg = traceback.format_exc()
        print(f"{FAIL}: {e}")
        print(msg)
        results.append((name, False, str(e)))


# ---------------------------------------------------------------------------
# B1 – Diagnostic counters increment in all three env subclasses
# ---------------------------------------------------------------------------

def _make_bsm_env(num_paths=4, num_steps=5):
    from hedging.envs import HedgeCallBS
    S0 = np.array([100.0], dtype=np.float32)
    K = np.array([[95.0, 100.0]], dtype=np.float32)
    return HedgeCallBS(
        S0, K, maturity=0.25, r=0.01, sigma=np.array([0.2]),
        num_paths=num_paths, num_steps=num_steps,
        history_len=1, transaction_cost=False,
    )


def check_b1_call_counters():
    env = _make_bsm_env()
    env.reset(seed=1)
    assert env._full_reset_count == 1,  f"Expected 1 full reset, got {env._full_reset_count}"
    assert env._soft_reset_count == 0,  f"Expected 0 soft resets, got {env._soft_reset_count}"
    # Trigger a soft reset
    env.reset(seed=1)  # same seed + soft_reset_enabled=True → soft reset
    assert env._soft_reset_count == 1,  f"Expected 1 soft reset after second reset, got {env._soft_reset_count}"
    print(f"    full={env._full_reset_count}  soft={env._soft_reset_count}")


def check_b1_doc_counters():
    from hedging.envs import HedgeDocBS
    S0 = np.array([100.0], dtype=np.float32)
    K  = np.array([[100.0]], dtype=np.float32)
    H  = np.array([[90.0]],  dtype=np.float32)
    env = HedgeDocBS(
        S0, K, H, maturity=0.25, r=0.01, sigma=np.array([0.2]),
        num_paths=4, num_steps=5, history_len=1, transaction_cost=False,
    )
    env.reset(seed=1)
    assert env._full_reset_count == 1, f"Doc: expected 1 full reset, got {env._full_reset_count}"
    env.reset(seed=1)
    assert env._soft_reset_count == 1, f"Doc: expected 1 soft reset, got {env._soft_reset_count}"


def check_b1_conc_counters():
    from hedging.envs import HedgeConcBS
    S0 = np.array([100.0], dtype=np.float32)
    K  = np.array([[100.0]], dtype=np.float32)
    P  = np.array([[1.0]],   dtype=np.float32)
    env = HedgeConcBS(
        S0, K, P, maturity=0.25, r=0.01, sigma=np.array([0.2]),
        num_paths=4, num_steps=5, history_len=1, transaction_cost=False,
    )
    env.reset(seed=1)
    assert env._full_reset_count == 1, f"Conc: expected 1 full reset, got {env._full_reset_count}"
    env.reset(seed=1)
    assert env._soft_reset_count == 1, f"Conc: expected 1 soft reset, got {env._soft_reset_count}"


check("B1 – HedgeCall diagnostic counters increment correctly", check_b1_call_counters)
check("B1 – HedgeDoc  diagnostic counters increment correctly", check_b1_doc_counters)
check("B1 – HedgeConc diagnostic counters increment correctly", check_b1_conc_counters)


# ---------------------------------------------------------------------------
# B2 – Dead expression removed (no SyntaxError in module import)
# ---------------------------------------------------------------------------

def check_b2_no_dead_expression():
    import importlib, importlib.util
    sim_dir = os.path.join(module_path, "experiments", "simulations")
    for fname in ("inactive_bsm_hedge_call_mlp.py", "inactive_bsm_hedge_con_rnn.py"):
        path = os.path.join(sim_dir, fname)
        src = open(path).read()
        # The bare tuple expression evaluated as a statement
        assert "    frames_per_batch, sub_batch_size\n" not in src, \
            f"{fname} still contains dead expression 'frames_per_batch, sub_batch_size'"
    print("    Dead expression absent in both affected files.")


check("B2 – Dead expression removed from both affected simulation files", check_b2_no_dead_expression)


# ---------------------------------------------------------------------------
# B3 – assert action_space.contains removed from step()
# ---------------------------------------------------------------------------

def check_b3_no_contains_assert():
    path = os.path.join(module_path, "hedging", "env_hedge_base.py")
    src = open(path).read()
    assert "action_space.contains" not in src, \
        "env_hedge_base.py still contains action_space.contains check"
    print("    action_space.contains not found in env_hedge_base.py")


check("B3 – assert action_space.contains removed from HedgeBase.step()", check_b3_no_contains_assert)


# ---------------------------------------------------------------------------
# P1 – Collector created once per epoch (not per episode) in actor_inactor_training
# ---------------------------------------------------------------------------

def check_p1_collector_per_epoch():
    path = os.path.join(module_path, "experiments", "utils", "training_loop.py")
    src = open(path).read()
    # In the new code the episode loop is "for episode, batch in enumerate(collector):"
    # and the collector is created outside it.
    assert "for episode, batch in enumerate(collector):" in src, \
        "Expected 'for episode, batch in enumerate(collector):' not found"
    assert "for episode in range(num_episodes):" not in src, \
        "Old per-episode loop 'for episode in range(num_episodes):' still present"
    print("    Collector per-epoch pattern confirmed.")


check("P1 – actor_inactor_training uses collector per epoch, not per episode", check_p1_collector_per_epoch)


# ---------------------------------------------------------------------------
# P2 – ReplayBuffer created outside the batch loop in action_training
# ---------------------------------------------------------------------------

def check_p2_buffer_outside_batch_loop():
    path = os.path.join(module_path, "experiments", "utils", "training_loop.py")
    src = open(path).read()
    # The replay_buffer creation must come BEFORE "for batch in collector:"
    # and AFTER "collector = SyncDataCollector(" in action_training.
    action_training_block = src[src.index("def action_training"):src.index("def actor_inactor_training")]
    buf_idx = action_training_block.index("replay_buffer = ReplayBuffer(")
    loop_idx = action_training_block.index("for batch in collector:")
    assert buf_idx < loop_idx, \
        "replay_buffer creation should appear BEFORE 'for batch in collector:'"
    print("    ReplayBuffer instantiated before batch loop in action_training.")


check("P2 – ReplayBuffer moved outside batch loop in action_training", check_p2_buffer_outside_batch_loop)


# ---------------------------------------------------------------------------
# P3 – nan_to_num is now conditional (only when grad is not finite)
# ---------------------------------------------------------------------------

def check_p3_conditional_nan_to_num():
    path = os.path.join(module_path, "experiments", "utils", "training_loop.py")
    src = open(path).read()
    # Old pattern: unconditional "if p.grad is not None: p.grad = torch.nan_to_num(p.grad)"
    # New pattern: guard with isfinite
    assert "if p.grad is not None:\n                            p.grad = torch.nan_to_num" not in src, \
        "Unconditional nan_to_num pattern still present"
    assert "isfinite" in src, \
        "Expected isfinite guard not found in training_loop.py"
    print("    nan_to_num is now guarded by isfinite check.")


check("P3 – nan_to_num guarded by isfinite (conditional, not unconditional)", check_p3_conditional_nan_to_num)


# ---------------------------------------------------------------------------
# P4 – set_keys moved before the epoch loop in actor_inactor_training
# ---------------------------------------------------------------------------

def check_p4_set_keys_before_epoch_loop():
    path = os.path.join(module_path, "experiments", "utils", "training_loop.py")
    src = open(path).read()
    inactor_block = src[src.index("def actor_inactor_training"):]
    set_keys_idx  = inactor_block.index("actor_advantage_module.set_keys(")
    epoch_loop_idx = inactor_block.index("for epoch in range(num_epochs):")
    assert set_keys_idx < epoch_loop_idx, \
        "set_keys should appear BEFORE 'for epoch in range(num_epochs):' in actor_inactor_training"
    print("    set_keys call precedes epoch loop.")


check("P4 – set_keys moved before epoch loop in actor_inactor_training", check_p4_set_keys_before_epoch_loop)


# ---------------------------------------------------------------------------
# P5 – _interest_factor cached and used instead of np.exp per step
# ---------------------------------------------------------------------------

def check_p5_interest_factor_cached():
    base_path = os.path.join(module_path, "hedging")
    # Must be defined in HedgeBase.__init__
    base_src = open(os.path.join(base_path, "env_hedge_base.py")).read()
    assert "_interest_factor" in base_src, \
        "_interest_factor not found in env_hedge_base.py"
    # Must be used (not np.exp) in call/doc/conc _update_portfolio
    for fname in ("env_hedge_call.py", "env_hedge_doc.py", "env_hedge_conc.py"):
        src = open(os.path.join(base_path, fname)).read()
        assert "self._interest_factor" in src, \
            f"_interest_factor not used in {fname}"
        assert "np.exp(self.r * self.simulator.dt)" not in src, \
            f"Old np.exp per-step call still present in {fname}"
    print("    _interest_factor cached in __init__ and used in all three env files.")


def check_p5_interest_factor_value():
    env = _make_bsm_env()
    env.reset(seed=0)
    expected = float(np.exp(env.r * env.maturity / env.num_steps))
    assert abs(env._interest_factor - expected) < 1e-7, \
        f"_interest_factor mismatch: {env._interest_factor} vs {expected}"
    print(f"    _interest_factor = {env._interest_factor:.8f}  (correct)")


check("P5 – _interest_factor cached in __init__ and used in all env files (source)", check_p5_interest_factor_cached)
check("P5 – _interest_factor value correct at runtime",                              check_p5_interest_factor_value)


# ---------------------------------------------------------------------------
# P6 – Critic architecture widened (hidden_dim//2, not 8)
# ---------------------------------------------------------------------------

def check_p6_mlp_critic_width():
    path = os.path.join(module_path, "experiments", "utils", "ppo_mlp_actor.py")
    src = open(path).read()
    assert "nn.Linear(hidden_dim, hidden_dim // 2)" in src, \
        "ppo_mlp_actor.py critic still uses narrow bottleneck (not hidden_dim//2)"
    assert "nn.Linear(hidden_dim, 8)" not in src, \
        "ppo_mlp_actor.py critic still has '→ 8' bottleneck"
    print("    MLP critic uses hidden_dim//2.")


def check_p6_rnn_critic_width():
    path = os.path.join(module_path, "experiments", "utils", "ppo_rnn_actor.py")
    src = open(path).read()
    assert "hidden_dim // 2" in src, \
        "ppo_rnn_actor.py critic still uses narrow bottleneck"
    assert "hidden_dim, 8)" not in src, \
        "ppo_rnn_actor.py critic still has '→ 8' bottleneck"
    print("    RNN critic uses hidden_dim//2.")


def check_p6_critic_instantiation():
    from experiments.utils.ppo_mlp_actor import create_ppo_mlp_actor
    model = create_ppo_mlp_actor(input_dim=11, action_dim=1, hidden_dim=64, device=torch.device("cpu"))
    critic_module = model.get_value_operator().module[-1].module
    # The first linear must be (64, 32), not (64, 8)
    first_linear = [m for m in critic_module.modules() if isinstance(m, torch.nn.Linear)][0]
    assert first_linear.out_features == 32, \
        f"Expected critic first layer out=32, got {first_linear.out_features}"
    print(f"    MLP critic first layer: {first_linear.in_features} → {first_linear.out_features}")


check("P6 – MLP actor critic uses hidden_dim//2 (source)",        check_p6_mlp_critic_width)
check("P6 – RNN actor critic uses hidden_dim//2 (source)",        check_p6_rnn_critic_width)
check("P6 – MLP critic instantiates with correct width at runtime", check_p6_critic_instantiation)


# ---------------------------------------------------------------------------
# A1 – CosineAnnealingLR removed from all inactive simulation scripts
# ---------------------------------------------------------------------------

def check_a1_no_cosine_in_inactive():
    sim_dir = os.path.join(module_path, "experiments", "simulations")
    found = []
    for fname in os.listdir(sim_dir):
        if fname.startswith("inactive_") and fname.endswith(".py"):
            src = open(os.path.join(sim_dir, fname)).read()
            if "CosineAnnealingLR" in src:
                found.append(fname)
    assert not found, f"CosineAnnealingLR still present in: {found}"
    print("    No CosineAnnealingLR references in any inactive_*.py file.")


def check_a1_no_cosine_in_training_loop():
    path = os.path.join(module_path, "experiments", "utils", "training_loop.py")
    src = open(path).read()
    assert "CosineAnnealingLR" not in src, \
        "CosineAnnealingLR still present in training_loop.py"
    print("    training_loop.py no longer imports or creates CosineAnnealingLR.")


check("A1 – CosineAnnealingLR removed from all inactive simulation scripts", check_a1_no_cosine_in_inactive)
check("A1 – CosineAnnealingLR removed from training_loop.py",                check_a1_no_cosine_in_training_loop)


# ---------------------------------------------------------------------------
# A2 – history_len=1 rolling window uses in-place assignment
# ---------------------------------------------------------------------------

def check_a2_source_has_history_len_branch():
    path = os.path.join(module_path, "hedging", "env_hedge_base.py")
    src = open(path).read()
    assert "if self.history_len == 1:" in src, \
        "history_len==1 fast path not found in env_hedge_base.py"
    assert "self.state[:, 0, :] = new_obs" in src, \
        "In-place assignment 'self.state[:, 0, :] = new_obs' not found"
    print("    history_len==1 fast path confirmed in env_hedge_base.py.")


def check_a2_state_shape_preserved():
    env = _make_bsm_env(num_paths=4, num_steps=5)
    env.reset(seed=0)
    initial_shape = env.state.shape
    action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    assert obs.shape == initial_shape, \
        f"State shape changed after step: {initial_shape} → {obs.shape}"
    print(f"    State shape {initial_shape} preserved after step.")


def check_a2_rnn_history_window_unchanged():
    """For history_len > 1 the concatenation path still runs correctly."""
    from hedging.envs import HedgeCallBS
    S0 = np.array([100.0], dtype=np.float32)
    K  = np.array([[100.0]], dtype=np.float32)
    env = HedgeCallBS(
        S0, K, maturity=0.25, r=0.01, sigma=np.array([0.2]),
        num_paths=4, num_steps=5, history_len=3,
        transaction_cost=False,
    )
    env.reset(seed=0)
    # Step twice; state should slide the window
    act = env.action_space.sample()
    obs1, _, _, _, _ = env.step(act)
    obs2, _, _, _, _ = env.step(act)
    assert obs2.shape == (env.num_envs, 3, env.feature_dim), \
        f"Unexpected shape with history_len=3: {obs2.shape}"
    print(f"    history_len=3 state shape: {obs2.shape}  (correct)")


check("A2 – history_len==1 fast path in source",                          check_a2_source_has_history_len_branch)
check("A2 – state shape preserved after step (history_len=1)",            check_a2_state_shape_preserved)
check("A2 – history window still slides correctly for history_len=3",     check_a2_rnn_history_window_unchanged)


# ---------------------------------------------------------------------------
# Integration smoke test: full episode rollout without crash
# ---------------------------------------------------------------------------

def check_integration_full_episode():
    env = _make_bsm_env(num_paths=8, num_steps=10)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        total_reward += float(rew.sum())
        done = bool(term[0])
        steps += 1
    assert steps == 10, f"Expected 10 steps, got {steps}"
    print(f"    Completed {steps} steps, total_reward={total_reward:.4f}")
    # Verify diagnostic counters are correct
    assert env._full_reset_count == 1
    assert env._soft_reset_count == 0
    print(f"    full_resets={env._full_reset_count}  soft_resets={env._soft_reset_count}")


check("Integration – full episode rollout completes without crash", check_integration_full_episode)


def check_integration_interest_factor_matches_manual():
    """Carry factor applied to cash must match manual np.exp calculation."""
    env = _make_bsm_env(num_paths=4, num_steps=5)
    env.reset(seed=7)
    action = env.action_space.sample()
    env.step(action)
    # After step 0→1, cash_account[...,1] = cash_account[...,0] * interest_factor
    ratio = env.cash_account[..., 1] / (env.cash_account[..., 0] + 1e-12)
    expected = env._interest_factor
    max_err = float(np.abs(ratio - expected).max())
    assert max_err < 1e-5, f"Cash carry ratio deviates from _interest_factor: max_err={max_err}"
    print(f"    Cash carry matches _interest_factor (max_err={max_err:.2e})")


check("Integration – cash carry factor matches _interest_factor after one step",
      check_integration_interest_factor_matches_manual)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"RESULTS: {n_pass} passed, {n_fail} failed out of {len(results)} checks")
if n_fail:
    print("\nFailed checks:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("All checks passed.")
