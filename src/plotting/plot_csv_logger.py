import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_csv_logger(
    csv_path: str,
    loss_names=["train_loss", "valid_loss"],
    eval_names=["train_mse", "valid_mse"],
):
    metrics = pd.read_csv(csv_path)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=True)

    ax1 = df_metrics[loss_names].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss", ax=ax1
    )

    ax2 = df_metrics[eval_names].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC", ax=ax2
    )
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    return df_metrics
