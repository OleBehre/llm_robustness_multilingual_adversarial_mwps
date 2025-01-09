import subprocess
import requests

# Command to be executed
command = [
    "CUDA_VISIBLE_DEVICES=7", 
    "lm_eval", 
    "--model", "hf", 
    "--include_path", "/home/obehre/gsm-eval/02_configurations/normal/", 
    "--device", "cuda", 
    "--batch_size", "4", 
    "--output_path", "/home/obehre/gsm-eval/results/"
]

languages = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "zh", "te", "th"]

models = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct", 
    "mistralai/Mistral-7B-Instruct-v0.3", 
    "mistralai/Mathstral-7B-v0.1,token=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",
    "LLM360/CrystalChat,trust_remote_code=True",
    "CohereForAI/aya-expanse-8b",
    ]

for m in models:
    for l in languages:
        print(f"Benchmarking language {l} on {m}")
        s = " ".join(command)
        s += f" --tasks gsm-normal-{l}" 
        s += f" --model_args pretrained={m}"

        result = subprocess.run(s, shell=True)
       