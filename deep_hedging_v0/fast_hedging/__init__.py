"""Fast, fully vectorised deep-hedging engine (PyTorch only, no Gym/TorchRL loop).

The market (paths and all instrument prices) is simulated and priced ONCE per
split and cached to disk. Training then samples paths from that bank and runs
the hedging recursion as batched tensor ops on the GPU.
"""
