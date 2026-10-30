from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_portfolio_vs_option_price(env, max_paths: int = 5, max_cols: int = 6):
    """Plot portfolio vs option price for a capped subset of paths and asset/strike combos."""

    S0, K = env.S0, env.K

    # Cap rows (paths) and cols (asset × strike combinations) to keep the plot readable
    show_paths   = min(env.num_paths, max_paths)
    show_cols    = min(env.num_assets * env.num_strikes, max_cols)

    # Build a flat list of (asset_idx, strike_idx) pairs up to show_cols
    col_pairs = [
        (i, j)
        for i in range(env.num_assets)
        for j in range(env.num_strikes)
    ][:show_cols]

    max_v_spacing = (1.0 / (show_paths - 1)) * 0.9 if show_paths > 1 else 0.05
    max_h_spacing = (1.0 / (show_cols - 1))  * 0.9 if show_cols  > 1 else 0.05

    fig = make_subplots(
        rows=show_paths,
        cols=show_cols,
        subplot_titles=[
            f"Asset {S0[i]:.0f}, K {K[i, j]:.0f}"
            for (i, j) in col_pairs
        ],
        shared_xaxes=True,
        vertical_spacing=min(0.05, max_v_spacing),
        horizontal_spacing=min(0.03, max_h_spacing),
    )

    # Add traces for each subplot
    for sim in range(show_paths):
        for col_pos, (i, j) in enumerate(col_pairs):
                # Calculate column index
                col_idx = col_pos + 1

                # Portfolio value line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(env.portfolio_value.shape[3])),
                        y=env.portfolio_value[sim, i, j, :],
                        mode="lines",
                        name=(
                            "Portfolio Value"
                            if sim == 0 and col_pos == 0
                            else None
                        ),
                        line=dict(color="blue", dash="solid"),
                        showlegend=(sim == 0 and col_pos == 0),
                    ),
                    row=sim + 1,
                    col=col_idx,
                )

                # Option price line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(env.option_prices.shape[3])),
                        y=env.option_prices[sim, i, j, :],
                        mode="lines",
                        name="Option Price" if sim == 0 and col_pos == 0 else None,
                        line=dict(color="red", dash="dash"),
                        showlegend=(sim == 0 and col_pos == 0),
                    ),
                    row=sim + 1,
                    col=col_idx,
                )

    # Update layout
    fig.update_layout(
        height=max(400, 200 * show_paths),
        width=max(600, 250 * show_cols),
        title_text="Portfolio Value vs Option Price (sample: "
                   f"{show_paths} paths × {show_cols} asset/strike combos)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update shared x and y axes titles
    fig.update_xaxes(title_text="Time Step")
    fig.update_yaxes(title_text="Value")

    # Show the figure
    fig.show()
