import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from transformers import AutoTokenizer

# Updated models list to include only "aya" and "gemma"
models = [
    #"google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct", 
    #"mistralai/Mistral-7B-Instruct-v0.3", 
    #"mistralai/Mathstral-7B-v0.1",
    "LLM360/CrystalChat",
   # "CohereForAI/aya-expanse-8b"
]


languages = ["en", "de", "fr", "es", "ru", "zh", "ja", "th", "te", "bn", "sw"]

# Prepare to hold all DataFrames for plotting later
all_data = []

# Function to calculate character lengths and prepare data for plotting
def calc(languages, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_data = []

    for l in languages:
        # File path to your JSONL file
        file_path = f'/home/obehre/gsm-eval/01_dataset/dataset_usable/test_normal/test-normal-{l}.jsonl'

        # List to store the processed data
        data = []

        # Open the JSONL file and process each line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                # Parse JSON line
                record = json.loads(line)
                answer = record.get("answer", "")

                # Calculate the number of tokens in "answer"
                num_tokens = len(tokenizer.encode(answer))
                print(f'{model} | {l} | {line_num}', end='\r')

                # Append data as a tuple
                data.append((line_num, num_tokens, l, model))

        # Create a DataFrame for this language
        df = pd.DataFrame(data, columns=['Line Number', 'Number of Tokens', 'Language', 'Model'])
        model_data.append(df)  

    # Combine all DataFrames for the current model
    combined_df = pd.concat(model_data)
    all_data.append(combined_df)

# Uncomment the following lines if you need to regenerate the CSV
# for m in models:
#     calc(languages, m)

# Combine all data into a single DataFrame
# final_df = pd.concat(all_data)
# final_df.to_csv('/home/obehre/gsm-eval/01_dataset/dataset_analysis/token_count_analysis.csv', index=False)
 
final_df = pd.read_csv('/home/obehre/gsm-eval/01_dataset/dataset_analysis/token_count_analysis.csv')

# Create a 2x1 subplot layout
fig, axes = plt.subplots(2, 1, figsize=(36, 20), sharey=True)
axes = axes.flatten()

# Set the color palette for the languages
colors = sns.color_palette("husl", len(languages))
color_map = {lang: colors[i] for i, lang in enumerate(languages)}

# Plot data for each model
for i, model in enumerate(models):
    model_df = final_df[final_df['Model'] == model]

    # Plot KDE for each language
    for lang in languages:
        lang_data = model_df.loc[model_df['Language'] == lang, 'Number of Tokens']
        sns.kdeplot(
            data=lang_data,
            label=None,  # Suppress automatic legend creation
            color=color_map[lang],
            ax=axes[i],
            linewidth=2,
            fill=True,   # Fill the area under the curve
            alpha=0.05    # Set transparency for the filled area
        )

        # Add vertical line for the 80th percentile
        percentile_value = lang_data.quantile(0.8)
        axes[i].axvline(
            x=percentile_value,
            color=color_map[lang],
            linestyle='--',
            linewidth=3
        )

    # Add titles and labels
    axes[i].set_title(model, fontsize=30)
    axes[i].set_xlabel('Number of Tokens', fontsize=30)
    axes[i].set_ylabel('Density', fontsize=30)
    axes[i].set_xlim(0, 850)
    axes[i].tick_params(axis='x', labelsize=30)  # Increase x-tick label size
    axes[i].tick_params(axis='y', labelsize=30)  # Increase y-tick label size
    axes[i].grid(True)

    # Add a legend to each subplot
    handles = [Patch(facecolor=color_map[lang], edgecolor='black', label=lang.upper()) for lang in languages]
    axes[i].legend(handles=handles, loc='upper right', fontsize=30, title="Languages", title_fontsize=18, handlelength=2, handleheight=1, handletextpad=0.5)


# Hide any unused subplots (if any)
for j in range(len(models), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Save the combined plot to a file
output_image = '/home/obehre/gsm-eval/05_results_evaluation/charts-tokenization/combined_analysis_llama_crystal.pdf' 
plt.savefig(output_image, dpi=150, bbox_inches='tight')
plt.close()
