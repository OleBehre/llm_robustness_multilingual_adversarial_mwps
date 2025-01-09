import pandas as pd
import matplotlib.pyplot as plt

# NOT USED 

# Load the CSV data
file_path = '/home/obehre/gsm-eval/05_results_evaluation/normal.csv' # Update with the correct file path
data = pd.read_csv(file_path)
# Define the desired model order
model_order = ['Gemma-2-9b-it', 'Llama-3.1-8B-Instruct', 'aya-expanse-8b', 'Mathstral-7B-v0.1', 'Mistral-7B-Instruct-v0.3', 'CrystalChat']
# Define the desired language order
language_order = ['en', 'de', 'fr', 'es', 'ru', 'zh', 'ja', 'th', 'te', 'bn', 'sw']

# Create a figure with 2x3 subplots
rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(36, 20), sharey=True)  # High-resolution figure size

axes = axes.flatten()

for i, model in enumerate(model_order):
    model_data = data[data['Model'] == model]
    model_data = model_data.set_index('Language').reindex(language_order).reset_index()  # Reorder languages
    languages = model_data['Language'].str.upper()
    accuracies = model_data['Accuracy']

    # Plot each model's data in its subplot
    bars = axes[i].bar(languages, accuracies, color='#4A90E2', alpha=0.6, edgecolor='#003366', linewidth=1.5)  # Transparent inner with darker outline
    axes[i].set_title(model, fontsize=30)
    axes[i].set_xlabel('Languages', fontsize=20)
    axes[i].set_ylabel('Accuracy', fontsize=20)
    axes[i].tick_params(axis='x', labelsize=20,)
    axes[i].tick_params(axis='y', labelsize=20)

# Hide any unused subplots
for j in range(len(model_order), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Save the combined plot to a file with high resolution
output_path = '/home/obehre/gsm-eval/05_results_evaluation/images/models_multilingual.pdf'  # Update with your desired directory
plt.savefig(output_path, dpi=150)  # High resolution
plt.close()
