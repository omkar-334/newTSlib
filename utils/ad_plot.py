import matplotlib.pyplot as plt
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


def plot_combined_anomaly_view(
    time_series_data,
    anomaly_scores,
    threshold,
    pred_mask,
    true_mask,
    save_path=None,
    title="Anomaly Detection Summary",
    feature_index=None,
    figsize=(15, 5),
    colors={"pred": "blue", "true": "yellow", "signal": "red"},
):
    """
    Plots anomaly scores with threshold, ground truth, and predicted anomaly regions,
    optionally overlaying a single feature from the original time series.

    Parameters
    ----------
    time_series_data : np.array
        (time_steps, n_features) time series data.
    anomaly_scores : np.array
        1D anomaly score array.
    threshold : float
        Threshold for anomaly detection.
    pred_mask : np.array
        Binary predicted anomaly mask.
    true_mask : np.array
        Binary ground truth anomaly mask.
    save_path : str
        Path to save the figure.
    title : str
        Title of the plot.
    feature_index : int
        Optional: index of the feature to overlay (default: average if None).
    figsize : tuple
        Figure size.
    colors : dict
        Dictionary of colors for "pred", "true", and "signal".
    """
    time_axis = np.arange(len(anomaly_scores))

    # Determine signal to plot
    if feature_index is not None:
        signal = time_series_data[:, feature_index]
    else:
        signal = time_series_data.mean(axis=1)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot signal
    ax1.plot(time_axis, signal, color=colors["signal"], alpha=0.4, label="Signal")

    # # Anomaly score
    ax2 = ax1.twinx()
    ax2.plot(
        time_axis, anomaly_scores, color="black", label="Anomaly Score", linewidth=1.2
    )
    ax2.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )

    # Shaded regions
    pred_regions = get_anomaly_regions(pred_mask)
    true_regions = get_anomaly_regions(true_mask)

    for start, end in true_regions:
        ax1.axvspan(start, end, color=colors["true"], alpha=0.2, label="Ground Truth")

    # for start, end in pred_regions:
    #     ax1.axvspan(start, end, color=colors["pred"], alpha=0.3, label="Predicted")

    # Titles and legends
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Signal")
    # ax2.set_ylabel("Anomaly Score")
    fig.suptitle(title, fontsize=14)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, handles1 + handles2))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Combined anomaly plot saved to: {save_path}")

    plt.show()
    plt.close()


def get_anomaly_regions(anomaly_mask):
    """
    Convert binary anomaly mask to list of (start, end) indices for continuous regions.

    Parameters
    ----------
    anomaly_mask : np.array
        Binary mask where 1 indicates anomaly

    Returns
    -------
    list of tuples
        Each tuple contains (start_idx, end_idx) of anomaly regions
    """
    regions = []
    if len(anomaly_mask) == 0:
        return regions

    # Find transitions
    diff = np.diff(np.concatenate(([0], anomaly_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        regions.append((start, end))

    return regions


def plot_multivariate_anomalies(
    time_series_data,
    anomaly_mask,
    feature_names=None,
    title="Multivariate Time Series Anomaly Detection",
    save_path=None,
    figsize=(15, 8),
    show_legend=True,
    alpha_shade=0.3,
    anomaly_color="red",
):
    """
    Plot multivariate time series with shaded anomaly regions.

    Parameters
    ----------
    time_series_data : np.array
        Shape (time_steps, n_features) - your multivariate time series
    anomaly_mask : np.array
        Shape (time_steps,) - binary mask where 1 indicates anomaly
    feature_names : list, optional
        Names for each feature/variable
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    show_legend : bool
        Whether to show legend
    alpha_shade : float
        Transparency of anomaly shading (0-1)
    anomaly_color : str
        Color for anomaly regions
    """
    # Ensure inputs are numpy arrays
    time_series_data = np.array(time_series_data)
    anomaly_mask = np.array(anomaly_mask)

    # Handle different input shapes
    if time_series_data.ndim == 1:
        time_series_data = time_series_data.reshape(-1, 1)

    n_timesteps, n_features = time_series_data.shape

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(n_features)]

    # Time axis
    time_axis = np.arange(n_timesteps)

    # Create subplots
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    if n_features == 1:
        axes = [axes]

    # Color palette for different features
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for i, ax in enumerate(axes):
        # Plot the time series
        ax.plot(
            time_axis,
            time_series_data[:, i],
            color=colors[i],
            linewidth=1.5,
            label=feature_names[i],
        )

        # Add shaded regions for anomalies
        anomaly_regions = get_anomaly_regions(anomaly_mask)
        for start_idx, end_idx in anomaly_regions:
            ax.axvspan(
                start_idx,
                end_idx,
                alpha=alpha_shade,
                color=anomaly_color,
                label="Anomaly" if i == 0 else "",
            )

        # Styling
        ax.set_ylabel(feature_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

        # Add legend only to first subplot if requested
        if show_legend and i == 0:
            ax.legend(loc="upper right", fontsize=9)

    # Set x-label only on bottom subplot
    axes[-1].set_xlabel("Time Steps", fontsize=11)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close()


def plot_anomaly_scores_with_threshold(
    anomaly_scores,
    threshold,
    anomaly_mask=None,
    title="Anomaly Scores with Threshold",
    save_path=None,
    figsize=(12, 4),
):
    """
    Plot anomaly scores with threshold line and detected anomalies.

    Parameters
    ----------
    anomaly_scores : np.array
        Anomaly scores for each time step
    threshold : float
        Threshold value for anomaly detection
    anomaly_mask : np.array, optional
        Binary mask of detected anomalies
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)

    time_axis = np.arange(len(anomaly_scores))

    # Plot anomaly scores
    plt.plot(
        time_axis,
        anomaly_scores,
        color="blue",
        linewidth=1.5,
        label="Anomaly Score",
        alpha=0.8,
    )

    # Plot threshold line
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.3f})",
    )

    # Shade anomaly regions if mask provided
    if anomaly_mask is not None:
        anomaly_regions = get_anomaly_regions(anomaly_mask)
        for start_idx, end_idx in anomaly_regions:
            plt.axvspan(start_idx, end_idx, alpha=0.3, color="red")

    plt.xlabel("Time Steps")
    plt.ylabel("Anomaly Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Anomaly scores plot saved to: {save_path}")

    plt.show()
    plt.close()


def create_comparison_plot(data, gt, pred, feature_names, save_path, setting):
    """Create a side-by-side comparison of ground truth vs predictions"""
    n_features = data.shape[1]
    fig, axes = plt.subplots(n_features, 2, figsize=(20, 4 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    time_axis = np.arange(len(data))
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for i in range(n_features):
        # Ground truth plot
        axes[i, 0].plot(time_axis, data[:, i], color=colors[i], linewidth=1.5)
        gt_regions = get_anomaly_regions(gt)
        for start_idx, end_idx in gt_regions:
            axes[i, 0].axvspan(start_idx, end_idx, alpha=0.3, color="orange")
        axes[i, 0].set_title(f"{feature_names[i]} - Ground Truth")
        axes[i, 0].grid(True, alpha=0.3)

        # Prediction plot
        axes[i, 1].plot(time_axis, data[:, i], color=colors[i], linewidth=1.5)
        pred_regions = get_anomaly_regions(pred)
        for start_idx, end_idx in pred_regions:
            axes[i, 1].axvspan(start_idx, end_idx, alpha=0.3, color="red")
        axes[i, 1].set_title(f"{feature_names[i]} - Predictions")
        axes[i, 1].grid(True, alpha=0.3)

    # Set x-labels for bottom row
    axes[-1, 0].set_xlabel("Time Steps")
    axes[-1, 1].set_xlabel("Time Steps")

    plt.suptitle(f"Anomaly Detection Comparison - {setting}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_anomaly_trace(
    signal, anomaly_mask=None, title="Anomaly Score", save_path=None
):
    """
    Minimal plot style for time series anomaly detection.

    signal: 1D array (anomaly scores or signal)
    anomaly_mask: 1D binary array, 1 where anomaly is detected
    title: string, title of the subplot
    save_path: optional path to save the figure
    """
    plt.figure(figsize=(2.2, 1.5))  # Small format
    plt.plot(signal, color="red", linewidth=1)

    if anomaly_mask is not None:
        for i in range(len(signal)):
            if anomaly_mask[i]:
                plt.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.1)

    plt.title(title, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
