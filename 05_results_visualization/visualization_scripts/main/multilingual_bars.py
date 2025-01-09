import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load data ---
file_path = '/home/obehre/gsm-eval/04_results/normal.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# --- 2. Define desired orders ---
model_order = [
    'Gemma-2-9b-it', 
    'Llama-3.1-8B-Instruct', 
    'aya-expanse-8b', 
    'Mathstral-7B-v0.1', 
    'Mistral-7B-Instruct-v0.3', 
    'CrystalChat'
]
language_order = ['en', 'de', 'fr', 'es', 'ru', 'zh', 'ja', 'th', 'te', 'bn', 'sw']

# --- 3. Pivot data ---
#    Rows: Language
#    Columns: Model
#    Values: Accuracy
df_pivot = data.pivot(
    index='Language', 
    columns='Model', 
    values='Accuracy'
)

# Ensure the pivoted DataFrame follows the specified orders
df_pivot = df_pivot.reindex(index=language_order, columns=model_order)

# Convert language codes to uppercase (for display on x-axis)
df_pivot.index = df_pivot.index.str.upper()

# --- 4. Plot as grouped bar chart ---
plt.figure(figsize=(18, 10))  # Adjust figure size as needed

# Optional: specify a color palette for the 6 models
color_palette = sns.color_palette("muted", n_colors=len(model_order))

ax = df_pivot.plot(
    kind='bar', 
    color=color_palette, 
    width=0.8, 
    figsize=(20, 10)
)



# --- 5. Customize plot ---
ax.set_title("Accuracy by Language and Model", fontsize=30)
ax.set_xlabel("Languages", fontsize=20)
ax.set_ylabel("Accuracy", fontsize=20)
ax.tick_params(axis='x', labelsize=30, rotation=0)
ax.tick_params(axis='y', labelsize=24)

# Place the legend at the bottom
plt.legend(
    title="Models",
    fontsize=24,
    title_fontsize=18,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.4),
    ncol=len(model_order) / 2
)

plt.tight_layout()

# --- 6. Save the figure ---
output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/models_multilingual.pdf'
plt.savefig(output_path, dpi=150)

output_path = '/home/obehre/gsm-eval/05_results_visualization/charts-main/models_multilingual.png'
plt.savefig(output_path, dpi=150)

plt.close()
