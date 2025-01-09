import subprocess
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import requests
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Command to be executed
command = [
    "lm_eval", 
    "--model", "hf", 
    "--include_path", "/home/obehre/gsm-eval/02_configurations/decoding_strategies/adversarial/", 
    "--device", "cuda", 
    "--batch_size", "1", 
]

languages = ["en", "zh", "te", "de", "sw"]

models = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct"
]

for m in models:

    print("Beam Search 4 Beams")
    for l in languages:
        print(f"Benchmarking language {l}")
        s = " ".join(command)
        s += f" --tasks gsm-adv-{l}-blank" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs num_beams=4,do_sample=False"
        s += f" --output_path /home/obehre/gsm-eval/04_results/decoding_experiments/adversarial/{l}/bs_4"

        result = subprocess.run(s, shell=True) 

    print("Diverse Beam Search 4 Beams 2 Groups")
    for l in languages:
        print(f"Benchmarking language {l} on 4_2")
        s = " ".join(command)
        s += f" --tasks gsm-adv-{l}-blank" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs num_beams=4,num_beam_groups=2,do_sample=False,diversity_penalty=1.0"
        s += f" --output_path /home/obehre/gsm-eval/04_results/decoding_experiments/adversarial/{l}/dbs_4_2"

        result = subprocess.run(s, shell=True)

    print("############## Temperature 0.1 ##############")
    for l in languages:
        print(f"Benchmarking language {l}")
        s = " ".join(command)
        s += f" --tasks gsm-adv-{l}-blank" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs temperature=0.1,do_sample=True"
        s += f" --output_path /home/obehre/gsm-eval/04_results/decoding_experiments/adversarial/{l}/temp_10"

        result = subprocess.run(s, shell=True)
        print("Done") 

    print("############## Temperature 0.7 ##############")
    for l in languages:
        print(f"Benchmarking language {l}")
        s = " ".join(command)
        s += f" --tasks gsm-adv-{l}-blank" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs temperature=0.7,do_sample=True"
        s += f" --output_path /home/obehre/gsm-eval/04_results/decoding_experiments/adversarial/{l}/temp_70"

        result = subprocess.run(s, shell=True)
        print("Done") 

    print("############## Top-K 5 ##############")
    for l in languages:
        print(f"Benchmarking language {l}")
        s = " ".join(command)
        s += f" --tasks gsm-adv-{l}-blank" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs temperature=1.0,top_k=5,do_sample=True"
        s += f" --output_path /home/obehre/gsm-eval/04_results/decoding_experiments/adversarial/{l}/top_k_5"

        result = subprocess.run(s, shell=True) 


    print("############## Top-p 0.3 ##############")
    for l in languages:
        print(f"Benchmarking language {l}")
        s = " ".join(command)
        s += f" --tasks gsm-adv-{l}-blank" 
        s += f" --model_args pretrained={m}"
        s += f" --gen_kwargs temperature=1.0,top_p=0.3,do_sample=True"
        s += f" --output_path /home/obehre/gsm-eval/04_results/decoding_experiments/adversarial/{l}/top_p_30"

        result = subprocess.run(s, shell=True)