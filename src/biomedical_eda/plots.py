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