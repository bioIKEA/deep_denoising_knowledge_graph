import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid", palette="pastel")

# Load the example tips dataset

data = pd.read_csv('boxplot.csv')
print(" data is", data)
# tips = sns.load_dataset("tips")
# print(" tips ", tips)
# Draw a nested boxplot to show bills by day and time
ax = sns.boxplot(x="Noise Ratio (NR)", y="AUC ROC Score",
            hue="Method", palette=["r", "b"],
            data=data)
# sns.despine(offset=10, trim=True)
plt.ylim(0.5, 1)
ax.set_xlabel('Noise Ratio (NR)', fontsize=10)
ax.set_ylabel('AUC ROC Score',fontsize=10)
plt.title('NetGAN vs CELL in different noise ratios', fontsize=12)
plt.show()
