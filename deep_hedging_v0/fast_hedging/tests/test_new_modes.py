"""Checks for the transformer encoder and the critic-free GRPO advantages (v2.4 probes).

  1. grpo_advantages equals a direct leave-one-out computation, is zero when every copy of a path
     gets the same rewards, and sums to zero over each group (norm='none').
  2. repeat_groups keeps the copies of a path together: a deterministic rollout gives the same P&L
     on every copy, and a stochastic one gives different P&L.
  3. the default config still builds the same window LSTM (no new parameters).
  4. exact resume (split run == whole run) for transformer and GRPO, single / band / double nets.
  5. the whole experiment under a tiny time budget, on the GPU when there is one, for the transformer
     with GRPO and the band net.

Usage (from deep_hedging_v0/):   python -m fast_hedging.tests.test_new_modes
"""
import sys
import tempfile

import torch

from fast_hedging.experiment import SPLIT_SEEDS
from fast_hedging.market import MarketSpec, get_bank
from fast_hedging.ppo import Agent, Task, TrainConfig, grpo_advantages, repeat_groups, rollout
from fast_hedging.tests.test_resume import SIZES, check_chain, check_exact


def check_grpo_math():
    g = torch.Generator().manual_seed(0)
    B, N, G, gamma = 12, 7, 4, 0.9
    r = torch.randn(B, N, generator=g)
    ret = torch.zeros(B, N)
    for t in range(N):
        ret[:, t] = sum(gamma ** (s - t) * r[:, s] for s in range(t, N))
    want = torch.zeros(B, N)
    for b in range(B):
        grp = range(b - b % G, b - b % G + G)
        want[b] = ret[b] - sum(ret[o] for o in grp if o != b) / (G - 1)
    got = grpo_advantages(r, gamma, G, norm="none")
    assert torch.allclose(got, want, atol=1e-5), "leave-one-out advantage differs from the direct computation"
    assert torch.allclose(got.view(B // G, G, N).sum(1), torch.zeros(B // G, N), atol=1e-5), "group sums are not zero"
    same = r.view(B // G, G, N)[:, :1].expand(B // G, G, N).reshape(B, N)
    assert grpo_advantages(same, gamma, G).abs().max() < 1e-4, "identical copies must get zero advantage"
    normed = grpo_advantages(r, gamma, G)
    sd = ret.view(B // G, G, N).std(1, keepdim=True)
    assert torch.allclose(normed.view(B // G, G, N), want.view(B // G, G, N) / sd, atol=1e-4), "group std scaling differs"
    print("grpo advantages OK", flush=True)


def check_groups():
    spec = MarketSpec(dynamics="bsm", product="call", market="sp500")
    bank = get_bank(spec, "train", SIZES["train"], SPLIT_SEEDS["train"], device="cpu")
    for arch in ("rnn", "transformer"):
        cfg = TrainConfig(arch=arch, net="band", advantage="grpo", grpo_group=3, window=10)
        task = Task(bank, "call", spec.r, cfg, "cpu")
        torch.manual_seed(0)
        agent = Agent(cfg, "call", "cpu")
        data = repeat_groups(task.batch(torch.arange(2)), 3)
        _, det = rollout(agent, task, data, use_inactor=False, actor_stochastic=False, record=False)
        p = det["pnl"].view(-1, 3)
        assert torch.allclose(p, p[:, :1].expand_as(p)), f"{arch}: copies of one path differ under a deterministic policy"
        _, sto = rollout(agent, task, data, use_inactor=False, actor_stochastic=True,
                         gen=torch.Generator().manual_seed(1), record=False)
        assert (sto["pnl"].view(-1, 3).std(1) > 0).all(), f"{arch}: stochastic copies are identical"
    print("grpo groups OK (rnn, transformer)", flush=True)


def check_default_unchanged():
    torch.manual_seed(0)
    agent = Agent(TrainConfig(window=10), "call", "cpu")
    names = [k for k, _ in agent.actor.named_parameters()]
    assert all(not k.startswith(("encoder.embed", "encoder.pos")) for k in names), names
    assert type(agent.actor.encoder.net).__name__ == "LSTM"
    n = sum(p.numel() for p in agent.actor.parameters())
    torch.manual_seed(0)
    t = Agent(TrainConfig(arch="transformer", window=10), "call", "cpu")
    nt = sum(p.numel() for p in t.actor.parameters())
    print(f"default net unchanged (window LSTM, {n} actor parameters; transformer {nt})", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_grpo_math()
    check_groups()
    check_default_unchanged()
    with tempfile.TemporaryDirectory() as tmp:
        for net, stops in (("single", [3]), ("band", [2, 5]), ("double", [3, 11])):
            check_exact(net, stops, tmp, arch="transformer", window=10)
            check_exact(net, stops, tmp, advantage="grpo", grpo_group=2)
        check_exact("band", [2, 5], tmp, arch="transformer", window=10, advantage="grpo", grpo_group=3)
        check_chain("band", tmp, ("--n-trials", "1", "--no-pruning", "--arch", "transformer", "--window", "10",
                                  "--advantage", "grpo", "--grpo-group", "2"))
    print("ALL PASSED")
    sys.exit(0)
