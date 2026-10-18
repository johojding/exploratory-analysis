import matplotlib.pyplot as plt

def plot_class_balance(class_balance):
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.bar(class_balance.index, class_balance["count"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Counts")
    ax.set_title("Class-count plot")
    return fig