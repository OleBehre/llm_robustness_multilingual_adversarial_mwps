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

#Call the calc function with all languages for all models
for m in models:
  calc(languages, m)

# Combine all data into a single DataFrame
final_df = pd.concat(all_data)
final_df.to_csv('/home/obehre/gsm-eval/01_dataset/dataset_analysis/token_count_analysis.csv', index=False)