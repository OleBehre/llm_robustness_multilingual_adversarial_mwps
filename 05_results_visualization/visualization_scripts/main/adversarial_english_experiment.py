import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
normal_file = '/home/obehre/gsm-eval/04_results/normal.csv'
adv_file = '/home/obehre/gsm-eval/04_results/adv.csv'

normal_data = pd.read_csv(normal_file)
adv_data = pd.read_csv(adv_file)

# Filter data for the language 'en'
normal_en = normal_data.loc[normal_data['Language'] == 'en'].copy()
adv_en = adv_data.loc[adv_data['Language'] == 'en'].copy()

# Ensure 'Model' columns are treated as strings and handle missing values
normal_en['Model'] = normal_en['Model'].astype(str).fillna('Unknown')
adv_en['Model'] = adv_en['Model'].astype(str).fillna('Unknown')

# Merge the data to compute differences
merged_en = pd.merge(normal_en, adv_en, on='Model', suffixes=('_normal', '_adv'))
merged_en['Accuracy_Difference'] = merged_en['Accuracy_adv'] / merged_en['Accuracy_normal']  # Reverse the subtraction

# Sort the data by the original model order for the first chart
model_order = ['Gemma-2-9b-it', 'Llama-3.1-8B-Instruct', 'aya-expanse-8b', 'Mathstral-7B-v0.1', 'Mistral-7B-Instruct-v0.3', 'CrystalChat']
merged_en['Model'] = pd.Categorical(merged_en['Model'], categories=model_order, ordered=True)
merged_en = merged_en.sort_values(by='Model')

# Create a sorted version for the difference chart
merged_en_diff_sorted = merged_en.sort_values(by='Accuracy_Difference', ascending=False)

# Plot the comparison and difference in accuracy
fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1]})

# Bar chart comparing normal and adversarial accuracies (side-by-side)
x = range(len(merged_en))  # Define x positions
width = 0.4  # Width of each bar
spacing = 0.05  # Tiny gap between bar groups

axes[0].bar([pos - width/2 - spacing/2 for pos in x], merged_en['Accuracy_normal'], width=width, color='#4A90E2', alpha=0.7, label='Normal', edgecolor='#003366', linewidth=1.5)
axes[0].bar([pos + width/2 + spacing/2 for pos in x], merged_en['Accuracy_adv'], width=width, color='#FFA500', alpha=0.7, label='Adversarial', edgecolor='#CC8400', linewidth=1.5)
axes[0].set_title('Accuracy Comparison for Language EN', fontsize=16)
axes[0].set_ylabel('Accuracy', fontsize=16)
axes[0].set_xticks(x)
axes[0].set_xticklabels(merged_en['Model'].astype(str), rotation=45, fontsize=14)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].legend()

# Bar chart showing the difference in accuracy
x_diff = range(len(merged_en_diff_sorted))
axes[1].bar(x_diff, merged_en_diff_sorted['Accuracy_Difference'], color='#FFA500', alpha=0.7, edgecolor='#CC8400', linewidth=1.5)  # Light orange color
axes[1].set_title('Accuracy Ratio (Adversarial / Normal)', fontsize=16)
axes[1].set_ylabel('Ratio', fontsize=16)
axes[1].set_xticks(x_diff)
axes[1].set_xticklabels(merged_en_diff_sorted['Model'].astype(str), rotation=45, fontsize=14)
axes[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()

#adjust the space between the two plots
plt.subplots_adjust(wspace=0.1)

# Save the plot
output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/en_accuracy_comparison_ratio.pdf'
plt.savefig(output_path, dpi=150)

output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/en_accuracy_comparison_ratio.png'
plt.savefig(output_path, dpi=150)
plt.close()
