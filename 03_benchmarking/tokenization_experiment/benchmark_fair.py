import subprocess
import requests
import pandas as pd

user_key = "uaanr77tp8a1pzjxjqm5vi8owtkzac"
api_token = "agnjq8byewo8t1vicm6qsz6maiq9um"
url = "https://api.pushover.net/1/messages.json"

# Command to be executed
command = [
    "HF_HOME=\"/work/obehre/hf-cache\"",
    "CUDA_VISIBLE_DEVICES=1", 
    "lm_eval", 
    "--model", "hf", 
    "--include_path", "/home/obehre/gsm-eval/02_configurations/tokenization_experiment/normal/", 
    "--device", "cuda", 
    "--batch_size", "4", 
    "--output_path", "/home/obehre/00_archive_resultsjsons/tokenizer_experiment/normal"
]

languages = ["bn", "en", "zh", "te", "th", "de", "es", "fr", "ja", "ru", "sw"]


models = [
    "meta-llama/Llama-3.1-8B-Instruct", 
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3", 
    "mistralai/Mathstral-7B-v0.1,token=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",
    "LLM360/CrystalChat,trust_remote_code=True"
    "CohereForAI/aya-expanse-8b"
    ]

for m in models:
    for l in languages:
        if m != "mistralai/Mathstral-7B-v0.1,token=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" and m != "LLM360/CrystalChat,trust_remote_code=True": 
            quantiles_file = f"/home/obehre/gsm-eval/01_dataset/dataset_analysis/quantiles/quantiles-{m.replace("/", "")}.csv"
        elif m is "LLM360/CrystalChat,trust_remote_code=True":
            quantiles_file = f"/home/obehre/gsm-eval/dataset_analysis/result/quantiles-LLM360CrystalChat.csv"
        else:
            quantiles_file = f"/home/obehre/gsm-eval/dataset_analysis/result/quantiles-mistralaiMathstral-7B-v0.1.csv"
            

        df = pd.read_csv(quantiles_file)
        max_gen_toks = int(df.loc[df["Language"] == l, "Number of Tokens"].iloc[0])
        

        print(f"Benchmarking language {l} on {m}")
        s = " ".join(command)
        s += f" --tasks gsm-fair-{l}" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs max_gen_toks={str(max_gen_toks)}"

        result = subprocess.run(s, shell=True)
        
    
      