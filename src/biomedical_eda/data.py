from biomedical_quality import quality_report, class_balance_report
from sklearn.datasets import load_breast_cancer


def load_dataset(): 
    dataset = load_breast_cancer(as_frame=True)

    df = dataset.data.copy()
    df["target"] = dataset.target
    return df, dataset.target_names


def generate_class_balance(df, target_names):
    class_balance = class_balance_report(df["target"], target_names)
    return class_balance


