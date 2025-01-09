import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

models = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct", 
    "mistralai/Mistral-7B-Instruct-v0.3", 
    "mistralai/Mathstral-7B-v0.1",
    "LLM360/CrystalChat",
    "CohereForAI/aya-expanse-8b"
]

languages = ["en", "de", "fr", "es", "ru", "zh", "ja", "th", "te", "bn", "sw"]
 
final_df = pd.read_csv('/home/obehre/gsm-eval/01_dataset/dataset_analysis/token_count_analysis.csv')

# Create a 2x3 subplot layout
fig, axes = plt.subplots(2, 3, figsize=(36, 20), sharey=True)
axes = axes.flatten()

# Plot data for each model
for i, model in enumerate(models):
    model_df = final_df[final_df['Model'] == model]

    # Set the color palette for the languages
    colors = sns.color_palette("husl", len(languages))
    color_map = {lang: colors[i] for i, lang in enumerate(languages)}

    # Plot KDE for each language
    for lang in languages:
        lang_data = model_df.loc[model_df['Language'] == lang, 'Number of Tokens']
        sns.kdeplot(
            data=lang_data,
            label=None,  # Suppress automatic legend creation
            color=color_map[lang],
            ax=axes[i],
            linewidth=2,
            fill=True,  # Fill the area under the curve
            alpha=0.05   # Set transparency for the filled area
        )

        # Add vertical line for the 80th percentile
        percentile_value = lang_data.quantile(0.8)
        axes[i].axvline(
            x=percentile_value,
            color=color_map[lang],
            linestyle='--'
        )

    # Add titles and labels
    axes[i].set_title(model, fontsize=30)
    axes[i].set_xlabel('Number of Tokens', fontsize=20)
    axes[i].set_ylabel('Density', fontsize=20)
    axes[i].set_xlim(0, 1000)
    axes[i].grid(True)

    # Add a legend to the bottom-right corner of the current subplot
    if i == 0:  # Add legend only to the first subplot
        handles = [plt.Line2D([], [], color=color_map[lang], lw=2, label=lang.upper()) for lang in languages]
        axes[i].legend(handles=handles, loc='upper right', fontsize=16, title="Languages", title_fontsize=18)

# Hide any unused subplots
for j in range(len(models), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Save the combined plot to a file
output_image = '/home/obehre/gsm-eval/05_results_evaluation/tokenization/combined_analysis_all_models_kde.pdf'
plt.savefig(output_image, dpi=150, bbox_inches='tight')
plt.close()
