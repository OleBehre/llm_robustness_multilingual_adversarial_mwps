import pandas as pd

# Read the CSV file
file_path = "/home/obehre/gsm-eval/04_results/adv.csv"  # Update this with the correct path to your CSV file
data = pd.read_csv(file_path)

# Pivot the data to create the table
table = data.pivot(index='Model', columns='Language', values='Accuracy')

# Convert accuracy values to percentages and round to one decimal place
table = (table * 100).round(1)

# Define the column order and ensure model order is preserved
columns_order = ['en', 'de', 'fr', 'es', 'ru', 'zh', 'ja', 'th', 'te', 'bn', 'sw']
model_order = ['Gemma-2-9b-it', 'Llama-3.1-8B-Instruct', 'Mathstral-7B-v0.1', 'Mistral-7B-Instruct-v0.3', 'CrystalChat', "aya-expanse-8b"]
table = table[columns_order].reindex(model_order)

# Generate the LaTeX table as a string
header = "\\begin{tabular}{l|c|cccccccccc}\\textbf{Normal, Strict} & \\textbf{EN} & \\textbf{DE} & \\textbf{FR} & \\textbf{ES} & \\textbf{RU} & \\textbf{ZH} & \\textbf{JA} & \\textbf{TH} & \\textbf{TE} & \\textbf{BN} & \\textbf{SW} \\\\ \\hline\n"
rows = []
for model, row in table.iterrows():
    row_values = " & ".join(row.fillna(" ").astype(str))
    rows.append(f"{model} & {row_values} \\\\")
body = "\n".join(rows)
footer = "\\end{tabular}"

latex_table = header + body + "\n" + footer

print(latex_table)
