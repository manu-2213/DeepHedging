# agent.py
import torch
import numpy as np
from model import TimeSeriesTransformer

class TransformerAgent:
    def __init__(self, model, device, num_asset, num_strike):
        self.model = model
        self.device = device
        self.num_asset = num_asset
        self.num_strike = num_strike

    def get_action(self, obs):
        """
        From the full state, extract a feature vector.
        (Here we assume the last 6 numbers in the state form the feature vector.)
        Then, run the transformer (as a sequence of length 1) to predict an action.
        """
        feat = obs[-6:]  # Assumed feature vector (length 6)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 6)
        # The model is set up so that output_dim = num_asset*num_strike.
        output = self.model(x)  # shape (1, 1, output_dim)
        output = output.squeeze(0).squeeze(0)  # shape (output_dim,)
        action = output.view(self.num_asset, self.num_strike).detach().cpu().numpy()
        return action
