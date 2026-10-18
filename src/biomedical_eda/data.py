from biomedical_quality import quality_report, class_balance_report
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)

df = dataset.data
targets = dataset.target
target_names = dataset.target_names

df_report, duplicates = quality_report(df)
df_targets = class_balance_report(targets, target_names)
