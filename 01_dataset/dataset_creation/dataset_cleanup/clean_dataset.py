# 1. UnicodeEscape needs to be turned into UTF-8
# 2. There should be no spaces between << and >> in the answer

import json
import re

def clean_adv_test_datasets(language):
    # Open the original JSONL file and create a new file with decoded content
    with open(f'/home/obehre/gsm-eval/dataset_translation/test_adv/test-adv-{language}.jsonl', 'r', encoding='utf-8') as infile, open(f'/home/obehre/gsm-eval/dataset_cleanup/test_adv/test-adv-{language}-clean.jsonl', 'w', encoding='utf-8') as outfile:
        for line_number, line in enumerate(infile, 1):  # Start line numbers from 1
           
            if line_number in invalid_rows:
                continue
            
            # Parse the JSON content
            data = json.loads(line)
            
            # 1. UnicodeEscape needs to be turned into UTF-8
            question = data['question'].encode('unicode_escape').decode('unicode_escape')
            answer = data['answer'].encode('unicode_escape').decode('unicode_escape')

            # 2. 
            answer = re.sub(r'<<(.*?)>>', lambda match: f"<<{match.group(1).replace(' ', '')}>>", answer)

            # 3.
            question = re.sub(r'(?<!\d)\.(?=[A-Za-z])', '. ', question)
            answer = re.sub(r'(?<!\d)\.(?=[A-Za-z])', '. ', answer)


            data['question'] = question 
            data['answer'] = answer


            # Write the decoded JSON data to the new file
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')  # Add a newline after each JSON object

def clean_normal_test_datasets(language):
    # Open the original JSONL file and create a new file with decoded content
    with open(f'/home/obehre/gsm-eval/dataset_creation/dataset_translation/test_normal/test-normal-{language}.jsonl', 'r', encoding='utf-8') as infile, open(f'/home/obehre/gsm-eval/dataset_creation/dataset_cleanup/test_normal/test-normal-{language}-clean-broken.jsonl', 'w', encoding='utf-8') as outfile:
         for line_number, line in enumerate(infile, 1):  # Start line numbers from 1
            
            if line_number in invalid_rows:
                continue
            

            # Parse the JSON content
            data = json.loads(line)
            
            # 1. UnicodeEscape needs to be turned into UTF-8
            question = data['question'].encode('unicode_escape').decode('unicode_escape')
            answer = data['answer'].encode('unicode_escape').decode('unicode_escape')

            # 2. 
            answer = re.sub(r'<<(.*?)>>', lambda match: f"<<{match.group(1).replace(' ', '')}>>", answer)

            # 3.
            question = re.sub(r'(?<!\d)\.(?=[A-Za-z])', '. ', question)
            answer = re.sub(r'(?<!\d)\.(?=[A-Za-z])', '. ', answer)

            data['question'] = question 
            data['answer'] = answer


            # Write the decoded JSON data to the new file
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')  # Add a newline after each JSON object

def clean_train_datasets(language):
    # Open the original JSONL file and create a new file with decoded content
    with open(f'/home/obehre/gsm-eval/dataset_creation/dataset_translation/train/train-{language}.jsonl', 'r', encoding='utf-8') as infile, open(f'/home/obehre/gsm-eval/dataset_creation/dataset_cleanup/train/train-{language}-clean.jsonl', 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Parse the JSON content
            data = json.loads(line)
            
            # 1. UnicodeEscape needs to be turned into UTF-8
            question = data['question'].encode('unicode_escape').decode('unicode_escape')
            answer = data['answer'].encode('unicode_escape').decode('unicode_escape')

            # 2. 
            answer = re.sub(r'<<(.*?)>>', lambda match: f"<<{match.group(1).replace(' ', '')}>>", answer)

            # 3.
            question = re.sub(r'(?<!\d)\.(?=[A-Za-z])', '. ', question)
            answer = re.sub(r'(?<!\d)\.(?=[A-Za-z])', '. ', answer)

            data['question'] = question 
            data['answer'] = answer


            # Write the decoded JSON data to the new file
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')  # Add a newline after each JSON object

def get_invalid_rows():
    invalid_lines = []
    
    # Open the JSONL file and check for invalid rows
    with open('/home/obehre/gsm-eval/dataset_creation/dataset_translation/test_adv/test-adv-en.jsonl', 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, 1):  # Start line numbers from 1
            # Parse the JSON content
            data = json.loads(line)
            
            # Check if the 'question' contains "NoInputFound" or "NoQuestionFound"
            if "NoInputFound" in data['question'] or "NoQuestionFound" in data['question']:
                invalid_lines.append(line_number)  # Append the line number
    
    return invalid_lines

langs = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

invalid_rows = get_invalid_rows()
print(invalid_rows)

for l in langs:
    clean_adv_test_datasets(l)
    clean_normal_test_datasets(l)
    clean_train_datasets(l)

    
  