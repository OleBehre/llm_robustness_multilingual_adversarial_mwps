import os
import json
import csv

# Define variables
noradv = ["normal", "adversarial"]
languages = ["de", "en", "sw", "te", "zh"]
strategies = ["bs_4", "dbs_4_2", "temp_10", "temp_70", "top_k_5", "top_p_30"]
models = ["google__gemma-2-9b-it", "meta-llama__Llama-3.1-8B-Instruct"]

# Specify base path
base_path = "/home/obehre/gsm-eval/04_results/decoding_experiments/"  # Replace with your base directory path

# Output CSV file path
output_csv_path = "/home/obehre/gsm-eval/04_results/decoding_experiments/results.csv"

# Prepare CSV file
with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(["model", "noradv", "language", "strategy", "strict_match_value"])

    # Iterate through all combinations
    for na in noradv:
        for la in languages:
            for stra in strategies:
                for mo in models:
                    # Construct the directory path
                    directory_path = f"{base_path}/{na}/{la}/{stra}/{mo}"
                    
                    try:
                        # Get the list of items in the directory
                        items = os.listdir(directory_path)
                        
                        # Find the first JSON file
                        json_file = next((item for item in items if item.endswith(".json")), None)
                        if json_file:
                            # Load the JSON data
                            with open(f"{directory_path}/{json_file}", "r", encoding="utf-8") as f:
                                data = json.load(f)
                                
                                # Extract the strict_match_value
                                key = f"gsm-{'normal' if na == 'normal' else 'adv'}-{la}-blank"
                                strict_match_value = data["results"].get(key, {}).get("exact_match,strict-match", None)
                                
                                # Write to CSV if the value exists
                                if strict_match_value is not None:
                                    writer.writerow([mo, na, la, stra, strict_match_value])
                    except Exception as e:
                        print(f"Error processing {directory_path}: {e}")
