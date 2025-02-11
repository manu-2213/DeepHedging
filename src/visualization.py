import matplotlib.pyplot as plt
import pandas as pd


# Define plotting of these functions

def plot_dataframe(df, title = "Price Evolution", x_label = "Time", y_label = "Price", show_labels = True):

    plt.figure(figsize=(10, 6))
    for col in df.columns[1:]:
        plt.plot(df["Time"], df[col], label=col)
    plt.xlabel(x_label)
    plt.ylabel("Price")
    plt.title(title)
    if show_labels:
        plt.legend()
    plt.grid(True)
    plt.show()