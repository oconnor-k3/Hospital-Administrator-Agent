import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_episode(history, config, path, title=""):
    """Queue by acuity, resource utilisation, and cumulative bad outcomes over one episode."""
    t = [h["t"] for h in history]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].stackplot(t, [h["queue_low"] for h in history], [h["queue_medium"] for h in history],
                    [h["queue_critical"] for h in history],
                    labels=["low", "medium", "critical"], colors=["#8fbf8f", "#e0b050", "#cc4444"])
    ax[0].set_ylabel("patients waiting")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, np.array([h["doctors_busy"] for h in history]) / config.n_doctors, label="doctors")
    ax[1].plot(t, np.array([h["nurses_busy"] for h in history]) / config.n_nurses, label="nurses")
    ax[1].plot(t, np.array([h["beds_busy"] for h in history]) / config.n_beds, label="beds")
    ax[1].set_ylabel("utilisation")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(loc="lower right")

    ax[2].plot(t, np.cumsum([h["deaths"] for h in history]), color="#cc4444", label="deaths")
    ax[2].plot(t, np.cumsum([h["deteriorated"] for h in history]), color="#e0b050", label="deteriorations")
    ax[2].plot(t, np.cumsum([h["discharged"] for h in history]), color="#4477aa", label="discharged")
    ax[2].set_ylabel("cumulative")
    ax[2].set_xlabel("tick (hour)")
    ax[2].legend(loc="upper left")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(results, metrics, path, title=""):
    """results: {policy: {metric: (mean, std)}}. One bar panel per metric."""
    names = list(results)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        means = [results[n][m][0] for n in names]
        stds = [results[n][m][1] for n in names]
        ax.bar(names, means, yerr=stds, capsize=3, color="#4477aa")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=30)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
