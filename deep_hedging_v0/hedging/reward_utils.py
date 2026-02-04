import numpy as np


def compute_discounted_cumsum_rewards(rewards, gamma):
    """
    Efficiently compute discounted returns for multiple games using matrix operations

    Parameters:
    - rewards: numpy array of shape (T, N) where T is number of time steps and N is number of games
    - gamma: discount factor

    Returns:
    - returns: numpy array of shape (T, N) containing discounted returns for each game
    """
    num_steps = rewards.shape[0]

    # Discount matrix construction
    indices = np.arange(num_steps)
    exponent_matrix = indices - indices[:, None]
    discount_matrix = np.triu(np.power(gamma, exponent_matrix))

    # Matrix multiplication to compute returns
    returns = discount_matrix @ rewards

    return returns
