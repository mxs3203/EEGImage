import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


# data = pd.read_csv("IG_res_summarize_channel.csv")
# annotation_bar = [row[-1] for row in data]
# plt.figure(figsize=(10, 6))
# sb.heatmap(data.drop(columns=['number', 'Unnamed: 0']), annot=False,linewidths=.5,
#            cmap="YlGnBu", cbar_kws={'label': 'Average Attribution'})
#
# plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')
#
#
# data = pd.read_csv("IG_res_per_channel.csv")
# grouped = data.drop(columns='number').groupby(['Unnamed: 0']).mean()
# max_values = grouped.max()
# plt.figure(figsize=(12, 6))
# sb.heatmap(grouped, annot=False,linewidths=.5,
#            cmap="YlGnBu", cbar_kws={'label': 'Average Attribution'})
# plt.savefig('theMostImportantTime.pdf', format='pdf', bbox_inches='tight')

sb.set_style("whitegrid")
data = pd.read_csv("IG_res_per_channel.csv")
grouped = data.drop(columns=['number']).groupby(['Unnamed: 0']).mean()
columns_to_color = ['EEG.Fp1', 'EEG.POz', 'EEG.FC4', 'EEG.F4', 'EEG.AF4', 'EEG.PO3']
for column in grouped.columns:
    if column in columns_to_color:
        plt.plot(grouped.index, grouped[column], marker='o', label=column)
    else:
        plt.plot(grouped.index, grouped[column], marker='o', linestyle='--', color='gray', alpha=0.2)

plt.title('Mean Values per Time Step')
plt.xlabel('Time Step')
plt.ylabel('Attribution Mean Value')
plt.grid(True)
plt.legend()
plt.savefig('theMostImportantTimeLinePlot.pdf', format='pdf', bbox_inches='tight')