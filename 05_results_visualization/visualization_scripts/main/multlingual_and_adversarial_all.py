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

# Initialize handles for the legend
handles = []
labels = ['Normal', 'Adversarial']

for i, model in enumerate(model_order):
    normal_model_data = normal_data[normal_data['Model'] == model]
    adversarial_model_data = adversarial_data[adversarial_data['Model'] == model]

    # Reindex to ensure languages are in the specified order
    normal_model_data = normal_model_data.set_index('Language').reindex(language_order).reset_index()
    adversarial_model_data = adversarial_data[adversarial_data['Model'] == model]
    adversarial_model_data = adversarial_model_data.set_index('Language').reindex(language_order).reset_index()

    languages = normal_model_data['Language'].str.upper()
    normal_accuracies = normal_model_data['Accuracy']
    adversarial_accuracies = adversarial_model_data['Accuracy']

    x = range(len(languages))
    width = 0.4

    # Plot normal and adversarial accuracies side-by-side
    normal_bars = axes[i].bar(
        [pos - width / 2 for pos in x],
        normal_accuracies,
        width=width,
        color='#4A90E2',
        alpha=0.6,
        edgecolor='#003366',
        linewidth=1.5,
        label='Normal'
    )
    adversarial_bars = axes[i].bar(
        [pos + width / 2 for pos in x],
        adversarial_accuracies,
        width=width,
        color='#FFA500',
        alpha=0.6,
        edgecolor='#CC8400',
        linewidth=1.5,
        label='Adversarial'
    )

    # Collect handles from the first subplot for the legend
    if i == 0:
        handles = [normal_bars, adversarial_bars]

    axes[i].set_title(model, fontsize=30)
    axes[i].set_xlabel('Languages', fontsize=24)
    axes[i].set_ylabel('Accuracy', fontsize=24)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(languages, fontsize=30)
    axes[i].tick_params(axis='y', labelsize=30)

    # Calculate AVG and STD for Normal and Adversarial
    normal_avg = normal_accuracies.mean()
    normal_std = normal_accuracies.std()
    adversarial_avg = adversarial_accuracies.mean()
    adversarial_std = adversarial_accuracies.std()

    #print(f"{model} + {adversarial_avg * 100} + {adversarial_std} \n")

    # Prepare the text to display
    stats_text = (
        f"Nor | AVG: {normal_avg:.2f} | STD: {normal_std:.2f}\n"
        f"Adv | AVG: {adversarial_avg:.2f} | STD: {adversarial_std:.2f}"
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

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leaves space at the bottom for the legend
plt.subplots_adjust(hspace=0.3, bottom=0.15)

# Save the combined plot to a file with high resolution
output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/models_multilingual_with_adversarial_avg_std.pdf'
plt.savefig(output_path, dpi=150, bbox_inches='tight')  

output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/models_multilingual_with_adversarial_avg_std.png' 
plt.savefig(output_path, dpi=150, bbox_inches='tight')  

plt.close()
