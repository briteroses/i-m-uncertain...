import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data
with open("linear_uncertainty_ccs.json", 'r') as file:
    data = json.load(file)

# using pandas df
data_list = []
for model, datasets in data.items():
    for dataset, metrics in datasets.items():
        metrics['model'] = model
        metrics['dataset'] = dataset
        data_list.append(metrics)
df = pd.DataFrame(data_list)

# accuracy differences
df['acc_diff'] = df['uccs_acc'] - df['ccs_acc']

# plot
sns.set_style("whitegrid")
plt.figure(figsize=(14, 10))

# red/green hue
palette = sns.diverging_palette(10, 133, as_cmap=True)

# scatterplot
scatter = sns.scatterplot(data=df, x="uccs_acc", y="uccs_coverage", hue="acc_diff", palette=palette,
                          style="model", s=100, edgecolor="black", linewidth=0.5)

# add the dataset names for each point
for i, row in df.iterrows():
    scatter.text(row["uccs_acc"] + 0.005, row["uccs_coverage"], row["dataset"], horizontalalignment='left',
                 verticalalignment='center', fontsize=10, color='darkgray')

# title, legend, labels for axis
scatter.set_title("Scatterplot of UCCS Accuracy vs. Coverage with Dataset Annotations")
scatter.legend(loc="upper left", title="Models")
scatter.set_xlabel("UCCS Accuracy")
scatter.set_ylabel("Coverage")
plt.tight_layout()

plt.show()
