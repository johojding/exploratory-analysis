import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

def plot_class_balance(class_balance: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.bar(class_balance.index, class_balance["count"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Counts")
    ax.set_title("Class-count plot")
    return fig

def plot_feature_histogram(df: pd.DataFrame, feature: str) -> Figure:
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.hist(df[feature])
    ax.set_xlabel(f"{feature.capitalize()}")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{feature.capitalize()} feature distribution")
    return fig

def plot_feature_by_class(df: pd.DataFrame, feature: str, target_names: np.ndarray) -> Figure:
    feature_df = df.loc[:, [feature, "target"]]
    class0 = feature_df[feature_df["target"] == 0]
    class1 = feature_df[feature_df["target"] == 1]

    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.boxplot([class0[feature], class1[feature]], tick_labels=target_names)
    ax.set_xlabel("Target class")
    ax.set_ylabel(feature.capitalize())
    ax.set_title(f"{feature.capitalize()} by class")
    return fig


def plot_scatter_by_class(df: pd.DataFrame, x_feature: str, y_feature: str, target_names: np.ndarray) -> Figure:
    feature_df = df.loc[:, [x_feature, y_feature, "target"]]
    class0 = feature_df[feature_df["target"] == 0]
    class1 = feature_df[feature_df["target"] == 1]

    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.scatter(class0[x_feature], class0[y_feature], color="blue", label=target_names[0])
    ax.scatter(class1[x_feature], class1[y_feature], color="orange", label=target_names[1])
    ax.set_xlabel(x_feature.capitalize())
    ax.set_ylabel(y_feature.capitalize())
    ax.set_title(f"{x_feature.capitalize()} vs {y_feature.capitalize()} scatter plot by class")
    ax.legend()
    return fig

def plot_correlation_heatmaps(df: pd.DataFrame, features: list[str], target_names: np.ndarray) -> Figure:
    nbr_features = len(features)
    features_df = df.loc[:, features + ["target"]]
    class0 = features_df[features_df["target"] == 0]
    class1 = features_df[features_df["target"] == 1]
    all_corr = features_df.loc[:, features].corr()
    class0_corr = class0.loc[:, features].corr()
    class1_corr = class1.loc[:, features].corr()

    fig, axs = plt.subplots(1, 3, figsize=(15,7), constrained_layout=True)

    im = axs[0].imshow(all_corr, vmin=-1, vmax=1, cmap="coolwarm")
    axs[0].set_xticks(range(nbr_features), labels=features, rotation=45)
    axs[0].set_yticks(range(nbr_features), labels=features, rotation=45)
    _add_heatmap_text(all_corr, axs[0])
    axs[0].set_title("Correlation heatmap all classes")

    axs[1].imshow(class0_corr, vmin=-1, vmax=1, cmap="coolwarm")
    axs[1].set_xticks(range(nbr_features), labels=features, rotation=45)
    axs[1].set_yticks(range(nbr_features), labels=features, rotation=45)
    _add_heatmap_text(class0_corr, axs[1])
    axs[1].set_title(f"Correlation heatmap class: {target_names[0]}")

    axs[2].imshow(class1_corr, vmin=-1, vmax=1, cmap="coolwarm")
    axs[2].set_xticks(range(nbr_features), labels=features, rotation=45)
    axs[2].set_yticks(range(nbr_features), labels=features, rotation=45)
    _add_heatmap_text(class1_corr, axs[2])
    axs[2].set_title(f"Correlation heatmap class: {target_names[1]}")

    fig.colorbar(im, ax=axs)
    return fig


def _add_heatmap_text(df, ax):
    for i in range(df.shape[0]):
            for j in range(df.shape[0]):
                value = df.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center")
