import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
normal_file_path = '/home/obehre/gsm-eval/04_results/normal.csv'
adversarial_file_path = '/home/obehre/gsm-eval/04_results/adv.csv'
normal_data = pd.read_csv(normal_file_path)
adversarial_data = pd.read_csv(adversarial_file_path)

# Define the desired model order
model_order = [
    'Gemma-2-9b-it', 
    'Llama-3.1-8B-Instruct', 
    'aya-expanse-8b', 
    'Mathstral-7B-v0.1', 
    'Mistral-7B-Instruct-v0.3', 
    'CrystalChat'
]

# Define the desired language order
language_order = ['en', 'de', 'fr', 'es', 'ru', 'zh', 'ja', 'th', 'te', 'bn', 'sw']

# Create a figure with 2x3 subplots
rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(36, 20), sharey=True)  # High-resolution figure size
axes = axes.flatten()

for i, model in enumerate(model_order):
    # Filter normal and adversarial data for this model
    normal_model_data = normal_data[normal_data['Model'] == model]
    adversarial_model_data = adversarial_data[adversarial_data['Model'] == model]

    # Reindex to ensure consistent language ordering
    normal_model_data = normal_model_data.set_index('Language').reindex(language_order).reset_index()
    adversarial_model_data = adversarial_model_data.set_index('Language').reindex(language_order).reset_index()

    languages = normal_model_data['Language'].str.upper()
    # Calculate ratio (adversarial / normal)
    ratios = adversarial_model_data['Accuracy'] / normal_model_data['Accuracy']

    x = range(len(languages))

    # Plot differences in accuracy
    axes[i].bar(x, ratios, color='#FFA500', alpha=0.6, edgecolor='#CC8400', linewidth=1.5)

    # Compute mean and standard deviation across all languages for this model
    mean_diff = ratios.mean()
    std_diff = ratios.std()

    # Remove the dashed line for the mean (previously axhline)

    # Annotate the mean and std in the top-right corner of each subplot
    text_x = 0.95
    text_y = 0.95
    axes[i].text(
        text_x, text_y,
        f"AVG: {mean_diff:.2f}\nSTD: {std_diff:.2f}",
        transform=axes[i].transAxes,
        fontsize=30,
        color='black',
        ha='right',          # align text to the right
        va='top'             # align text to the top
    )

    axes[i].set_title(model, fontsize=30)
    axes[i].set_xlabel('Languages', fontsize=24)
    axes[i].set_ylabel('Ratio', fontsize=24)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(languages, fontsize=30)
    axes[i].tick_params(axis='y', labelsize=30)

# Hide any unused subplots (in case you use fewer than 6 models)
for j in range(len(model_order), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Save the combined plot to a file with high resolution
output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/models_language_divided.pdf'  
plt.savefig(output_path, dpi=150)  

output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/models_language_divided.png'  
plt.savefig(output_path, dpi=150)  

plt.close()
