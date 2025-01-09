# Bengali (BN), 
# Chinese (ZH), 
# French (FR), 
# German (DE), 
# Japanese (JA), 
# Russian (RU), 
# Spanish (ES), 
# Swahili (SW), 
# Telugu (TE), 
# Thai (TH)

import argparse
import time
from translatepy.translators.google import GoogleTranslate
import json
import os


translator = GoogleTranslate()

def translate_dataset_to_language(language):
    print(f"Translating dataset to language {language}...")


    original_dataset_path = "/home/obehre/gsm-eval/dataset_translation/test_normal/test-normal-en.jsonl"
    new_dataset_path = f"/home/obehre/gsm-eval/dataset_translation/test_normal_translated/test-normal-{language}.jsonl"

    new_entry = {
        "question": "NewQ",
        "answer": "NewA"
    }

    #stop program if new dataset already exists
    if os.path.exists(new_dataset_path):
        print("Dataset with that name already exists.")
        exit()

    #save start time
    start = time.time()

    # for each entry in the original dataset add new_entry to the new one
    with open(original_dataset_path, 'r') as file:
        entry_count = 1
        warning_count = 0

        for line in file:
            
            retry_count = 0
            success = False

            while success == False:
                try:
                    print(f"Processing entry {entry_count} / 1318 | {entry_count / 1318 * 100:.2f}%", end="\r")
                
                    entry = json.loads(line)
                    new_entry["question"] = translator.translate(entry["question"], language).result
                    new_entry["answer"] = translator.translate(entry["answer"], language).result
                    
                    
                    with open(new_dataset_path, 'a') as file:
                        json.dump(new_entry, file)
                        file.write('\n')

                    success = True
                    entry_count += 1
                except Exception:
                    retry_count += 1

                    # Give up after trying the same entry 10 times
                    if(retry_count > 10):
                        print(f"ERROR: After trying to process entry {entry_count} / 1318 this script will be shut down.")
                        exit()

                    print(f"WARNING: Exception processing entry {entry_count} / 1318")
                    warning_count += 1
                    time.sleep(20)
                    print("Retrying...")
                
    print(f"Dataset successfully translated to language {language}.")
    print(f"Time taken: {time.time() - start} seconds")
    print(f"Warnings: {warning_count}")
    print("----------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate dataset to a specified language.")
    parser.add_argument("language", help="The language code.")
    args = parser.parse_args()

    translate_dataset_to_language(args.language)

    
    




