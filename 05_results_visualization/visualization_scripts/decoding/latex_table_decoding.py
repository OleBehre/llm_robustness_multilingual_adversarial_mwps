import pandas as pd


noradv = ["normal", "adversarial"]
languages = ["en", "de", "zh", "te", "sw"]
strategies = ["Greedy", "BS", "DBS", "T (0.1)", "T (0.7)", "Top-$k$", "Top-$p$"]
models = ["google__gemma-2-9b-it", "meta-llama__Llama-3.1-8B-Instruct"]


# Read the CSV file
file_path = "/home/obehre/gsm-eval/04_results/decoding.csv"  # Update this with the correct path to your CSV file
data = pd.read_csv(file_path)

only_model = data[data['model'].str.contains("meta-llama__Llama-3.1-8B-Instruct", case=False)]
#only_mode = only_model[only_model['mode'].str.contains("normal", case=False)]

# drop model column
only_model = only_model.drop(columns=['model'])



table = ""
# Generate the LaTeX table as a string
header = "\\begin{tabular}{ll|ccccc|cc} Strategy & Mode & \\textbf{EN} & \\textbf{DE} & \\textbf{ZH} & \\textbf{TE} & \\textbf{SW} & AVG & STD \\\\ \\hline\n"
footer = "\\end{tabular}"
for strat in strategies:
    c = 0
    for m in noradv:
        if c == 0:
            built_string = "\multirow{2}{*}{" + f"{strat}" + "}" + f" & {m} "
        else:
            built_string = f"& {m} "

        for l in languages:
            filtered_value = only_model[
                (only_model['language'] == l) &
                (only_model['strategy'] == strat) &
                (only_model['mode'] == m)
            ]
            value = filtered_value["accuracy"].values[0] * 100
            built_string += f"& {value.round(2)} "
        
        filtered_accuracy = only_model[
            (only_model['strategy'] == strat) &
            (only_model['mode'] == m)
        ]

        table += built_string + f"& {(filtered_accuracy['accuracy'].mean() * 100).round(2)} &  {(filtered_accuracy['accuracy'].std() * 100).round(2)} \\\\" 
        
        
     
        

        if c == 0:
            table += " \n"
        else:
            table += " \\hline\n"
             
        c = 1
print(header + table + footer)


print()