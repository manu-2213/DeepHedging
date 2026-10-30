from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_portfolio_vs_option_price(env):

    num_simulation, num_asset, num_strike = (
        env.num_simulation,
        env.num_asset,
        env.num_strike,
    )

    S0, K = env.S0, env.K

    fig = make_subplots(
        rows=num_simulation,
        cols=num_asset * num_strike,
        subplot_titles=[
            f"Asset {S0[i]}, Strike {K[i, j]}"
            for sim in range(num_simulation)
            for i in range(len(S0))
            for j in range(K.shape[1])
        ][
            : num_asset * num_strike
        ],  # Only need titles for one simulation
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
    )

    # Add traces for each subplot
    for sim in range(num_simulation):
        for i in range(len(S0)):
            for j in range(K.shape[1]):
                # Calculate column index
                col_idx = i * K.shape[1] + j + 1

                # Portfolio value line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(env.portfolio_value.shape[3])),
                        y=env.portfolio_value[sim, i, j, :],
                        mode="lines",
                        name=(
                            "Portfolio Value"
                            if sim == 0 and i == 0 and j == 0
                            else None
                        ),
                        line=dict(color="blue", dash="solid"),
                        showlegend=(sim == 0 and i == 0 and j == 0),
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
                        name="Option Price" if sim == 0 and i == 0 and j == 0 else None,
                        line=dict(color="red", dash="dash"),
                        showlegend=(sim == 0 and i == 0 and j == 0),
                    ),
                    row=sim + 1,
                    col=col_idx,
                )

    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title_text="Portfolio Value vs Option Price Across Simulations",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update shared x and y axes titles
    fig.update_xaxes(title_text="Time Step")
    fig.update_yaxes(title_text="Value")

    # Show the figure
    fig.show()
