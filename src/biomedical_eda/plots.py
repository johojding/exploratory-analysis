import matplotlib.pyplot as plt
import pandas as pd

def plot_class_balance(class_balance: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.bar(class_balance.index, class_balance["count"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Counts")
    ax.set_title("Class-count plot")
    return fig

def plot_feature_histogram(df: pd.DataFrame, feature: str) -> plt.figure:
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.hist(df[feature])
    ax.set_xlabel(f"{feature.capitalize()}")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{feature.capitalize()} feature distribution")
    return fig