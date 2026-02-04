import numpy as np
# TO DO: add H observation for DOC
# Define safe_log
EPS = 1e-8

def safe_log(x):
    return np.log(np.clip(a=x, a_min=EPS, a_max=None))


def create_observation_hedge_call(
    current_step,
    stock_prices,
    option_prices,
    K_expanded,
    r,
    T,
    num_strikes,
    num_total_options,
):

    step_idx = current_step
    # Get current stock prices
    S_current = stock_prices[:, :, step_idx]  # (num_simulation, num_asset)

    T_current = T[step_idx]

    # Expand stock price to match K dimensions
    S_expanded = S_current[:, :, None].repeat(
        num_strikes, axis=2
    )  # (num_simulation, num_asset, num_strike)

    # Calculate moneyness and other common features
    moneyness = K_expanded / S_expanded
    moneyness_log = safe_log(moneyness)
    time_feature = np.full_like(moneyness, T_current)
    risk_free_rate = np.full_like(moneyness, r)

    if step_idx == 0:
        # For initial step, set price_change to 1 and log_change to 0
        asset_price_change = np.zeros_like(moneyness)
        asset_price_change_ratio = np.ones_like(moneyness)
        asset_price_change_log = np.zeros_like(moneyness)
        option_price_change = np.zeros_like(moneyness)
        option_price_change_ratio = np.ones_like(moneyness)
        option_price_change_log = np.zeros_like(moneyness)
        est_delta = np.zeros_like(moneyness)
    else:
        # Calculate price changes for steps after the first one
        S_prev = stock_prices[:, :, step_idx - 1]
        S_prev_expanded = S_prev[:, :, None].repeat(num_strikes, axis=2)
        asset_price_change = S_expanded - S_prev_expanded
        asset_price_change_ratio = (S_expanded + np.finfo(S_expanded.dtype).eps) / (
            S_prev_expanded + np.finfo(S_expanded.dtype).eps
        )
        asset_price_change_log = safe_log(asset_price_change_ratio)
        option_price_change = (
            option_prices[:, :, :, step_idx] - option_prices[:, :, :, step_idx - 1]
        )
        option_price_change_ratio = (
            option_prices[:, :, :, step_idx] + np.finfo(option_prices.dtype).eps
        ) / (option_prices[:, :, :, step_idx - 1] + np.finfo(option_prices.dtype).eps)
        option_price_change_log = safe_log(option_price_change_ratio)
        est_delta = option_price_change / (
            asset_price_change + np.finfo(asset_price_change.dtype).eps
        )
        est_delta = np.clip(est_delta, 0.0, 1.0)

    # Stack features
    features = np.stack(
        [
            asset_price_change,  # Asset price change
            asset_price_change_ratio,  # Asset price change ratio
            asset_price_change_log,  # Asset price change (log)
            option_price_change,  # Option price change
            option_price_change_ratio,  # Option price change ratio
            option_price_change_log,  #  Option price change (log)
            est_delta,  # Estimated delta
            moneyness,  # Moneyness (K/S)
            moneyness_log,  # Log moneyness
            time_feature,  # Time to maturity
            risk_free_rate,  # Risk-free rate
        ],
        axis=-1,
    )

    # Reshape to have a flat batch dimension
    return features.reshape(num_total_options, -1)


def create_observation_hedge_doc(
    current_step,
    stock_prices,
    option_prices,
    call_prices,
    put_prices,
    K_expanded,
    r,
    T,
    num_strikes,
    num_total_options,
):
    features = create_observation_hedge_call(
        current_step,
        stock_prices,
        option_prices,
        K_expanded,
        r,
        T,
        num_strikes,
        num_total_options,
    )
    step_idx = current_step
    if step_idx == 0:
        # For initial step, set price_change to 1 and log_change to 0
        call_prices_changes = np.zeros_like(option_prices[..., 0])
        put_prices_change = np.zeros_like(option_prices[..., 0])
        call_prices_ratio = np.ones_like(option_prices[..., 0])
        put_prices_ratio = np.ones_like(option_prices[..., 0])
        call_prices_log = np.zeros_like(option_prices[..., 0])
        put_prices_log = np.zeros_like(option_prices[..., 0])
    else:
        call_prices_changes = (
            call_prices[..., step_idx] - call_prices[..., step_idx - 1]
        )
        put_prices_change = put_prices[..., step_idx] - put_prices[..., step_idx - 1]
        call_prices_ratio = (
            call_prices[..., step_idx] + np.finfo(call_prices.dtype).eps
        ) / (call_prices[..., step_idx - 1] + np.finfo(call_prices.dtype).eps)
        put_prices_ratio = (
            put_prices[..., step_idx] + np.finfo(put_prices.dtype).eps
        ) / (put_prices[..., step_idx - 1] + np.finfo(put_prices.dtype).eps)
        call_prices_log = safe_log(call_prices_ratio)
        put_prices_log = safe_log(put_prices_ratio)
    # Stack features
    features_doc = np.stack(
        [
            call_prices_changes,  # Call prices change
            put_prices_change,  # Put prices change
            call_prices_ratio,  # Call prices change ratio
            put_prices_ratio,  # Put prices change ratio
            call_prices_log,  # Call prices change (log)
            put_prices_log,  # Put prices change (log)
        ],
        axis=-1,
    )
    # Reshape to have a flat batch dimension
    features_doc = features_doc.reshape(num_total_options, -1)
    return np.concatenate((features, features_doc), axis=-1)  # Combine features

def create_observation_hedge_conc(
    current_step,
    stock_prices,
    option_prices, 
    call_prices,
    put_prices,
    K_expanded,
    r,
    T,
    num_strikes,
    num_total_options,
):
    features = create_observation_hedge_call(
        current_step,
        stock_prices,
        option_prices,
        K_expanded,
        r,
        T,
        num_strikes,
        num_total_options,
    )
    step_idx = current_step
    if step_idx == 0:
        # For initial step, set price_change to 1 and log_change to 0
        call_prices_changes = np.zeros_like(option_prices[..., 0])
        put_prices_change = np.zeros_like(option_prices[..., 0])
        call_prices_ratio = np.ones_like(option_prices[..., 0])
        put_prices_ratio = np.ones_like(option_prices[..., 0])
        call_prices_log = np.zeros_like(option_prices[..., 0])
        put_prices_log = np.zeros_like(option_prices[..., 0])
    else:
        call_prices_changes = (
            call_prices[..., step_idx] - call_prices[..., step_idx - 1]
        )
        put_prices_change = put_prices[..., step_idx] - put_prices[..., step_idx - 1]
        call_prices_ratio = (
            call_prices[..., step_idx] + np.finfo(call_prices.dtype).eps
        ) / (call_prices[..., step_idx - 1] + np.finfo(call_prices.dtype).eps)
        put_prices_ratio = (
            put_prices[..., step_idx] + np.finfo(put_prices.dtype).eps
        ) / (put_prices[..., step_idx - 1] + np.finfo(put_prices.dtype).eps)
        call_prices_log = safe_log(call_prices_ratio)
        put_prices_log = safe_log(put_prices_ratio)
    # Stack features
    features_doc = np.stack(
        [
            call_prices_changes,  # Call prices change
            put_prices_change,  # Put prices change
            call_prices_ratio,  # Call prices change ratio
            put_prices_ratio,  # Put prices change ratio
            call_prices_log,  # Call prices change (log)
            put_prices_log,  # Put prices change (log)
        ],
        axis=-1,
    )
    # Reshape to have a flat batch dimension
    features_doc = features_doc.reshape(num_total_options, -1)
    return np.concatenate((features, features_doc), axis=-1)  # Combine features
