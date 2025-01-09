import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
fair_file_path = '/home/obehre/gsm-eval/04_results/tokenization.csv'
normal_file_path = '/home/obehre/gsm-eval/04_results/normal.csv'

fair_data = pd.read_csv(fair_file_path)
normal_data = pd.read_csv(normal_file_path)

# Define the desired model order
model_order = ['Gemma-2-9b-it', 'Llama-3.1-8B-Instruct', 'aya-expanse-8b', 'Mathstral-7B-v0.1', 'Mistral-7B-Instruct-v0.3', 'CrystalChat']
# Define the desired language order
language_order = ['en', 'de', 'fr', 'es', 'ru', 'zh', 'ja', 'th', 'te', 'bn', 'sw']

# Create a figure with 2x3 subplots
rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(36, 20), sharey=True)  # High-resolution figure size

axes = axes.flatten()

handles = []
labels = ['Normal', 'Fair']

for i, model in enumerate(model_order):
    fair_model_data = fair_data[fair_data['Model'] == model]
    normal_model_data = normal_data[normal_data['Model'] == model]
    
    # Reorder languages for both datasets
    fair_model_data = fair_model_data.set_index('Language').reindex(language_order).reset_index()
    normal_model_data = normal_model_data.set_index('Language').reindex(language_order).reset_index()
    
    languages = fair_model_data['Language'].str.upper()
    fair_accuracies = fair_model_data['Accuracy']
    normal_accuracies = normal_model_data['Accuracy']

    # Plot each model's data in its subplot
    x = range(len(languages))
    width = 0.4  # Width of each bar
    
    # Plot normal accuracies
    normal_bars = axes[i].bar([p - width/2 for p in x], normal_accuracies, width=width, color='#4A90E2', alpha=0.8, edgecolor='#003366', linewidth=1.5, label='Normal')
    # Plot fair accuracies
    fair_bars = axes[i].bar([p + width/2 for p in x], fair_accuracies, width=width, color='#66B266', alpha=0.8, edgecolor='#004D00', linewidth=1.5, label='Fair')

    axes[i].set_title(model, fontsize=30)
    axes[i].set_xlabel('Languages', fontsize=24)
    axes[i].set_ylabel('Accuracy', fontsize=24)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(languages, fontsize=30)
    axes[i].tick_params(axis='y', labelsize=30)

    # Calculate AVG and STD for Normal and Adversarial
    normal_avg = normal_accuracies.mean()
    normal_std = normal_accuracies.std()
    fair_avg = fair_accuracies.mean()
    fair_std = fair_accuracies.std()

    print(f"Model: {model} & {fair_avg*100:.1f} & {fair_std:.2f}")

    if i == 0:
        handles = [normal_bars, fair_bars]

    # Prepare the text to display
    stats_text = (
        f"Nor | AVG: {normal_avg:.2f} | STD: {normal_std:.2f}\n"
        f"Fair | AVG: {fair_avg:.2f} | STD: {fair_std:.2f}"
    )

    # Add the text box to the top right corner
    # Adjust the coordinates (0.95, 0.95) as needed
    axes[i].text(
        0.95, 0.95, stats_text,
        transform=axes[i].transAxes,
        fontsize=30,
        verticalalignment='top',
        horizontalalignment='right',
        # bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )

# Hide any unused subplots
for j in range(len(model_order), len(axes)):
    axes[j].axis('off')

# Add a single legend at the bottom center
fig.legend(handles, labels, fontsize=40, loc='lower center', ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.subplots_adjust(hspace=0.3, bottom=0.15)  

# Save the combined plot to a file with high resolution
output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-tokenization/models_multilingual_fair_vs_normal_avg_std.png' 
plt.savefig(output_path, dpi=150) 

output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-tokenization/models_multilingual_fair_vs_normal_avg_std.pdf'
plt.savefig(output_path, dpi=150) 
plt.close()
