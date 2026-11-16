"""The hedging engine: a fully vectorised deep-hedging environment and learner (PyTorch only, no environment loop).

The market (paths and all instrument prices) is simulated and priced ONCE per split and cached to disk. Training then
samples paths from that bank and runs the hedging recursion as batched tensor operations on the GPU.

  calibration   calibrated Black-Scholes and Heston parameters per year, train / test split by year
  market        simulation, market banks and their checks, the bank cache
  pricing       vectorised Black-Scholes and Heston pricing, barrier Monte Carlo
  features      observations
  networks      encoders, actor, inaction network
  ppo           the hedging recursion (Task, rollout), PPO / GRPO updates, the training loop, evaluation, simple baselines
  model_band    model-based reference strategy: delta / delta-gamma target with a no-trade band

This package is a library. It contains no experiment and never imports the `experiments` package; a training run is
`python -m experiments.train`.

Comments that name `hedging/...`, `HedgeBase`, `BlackScholesSimulator`, `HestonSimulator` or "the old code" refer to
the original Gym/TorchRL implementation that this engine replaced and was validated against. That code is in the git
history (tag pre-cleanup-2026-11-16).
"""
