import json
import re

with open(f'/home/obehre/gsm-eval/dataset_creation/dataset_cleanup/test_normal/test-normal-en-clean.jsonl', 'r', encoding='utf-8') as enfile:
    with open(f'/home/obehre/gsm-eval/dataset_creation/dataset_cleanup/test_normal/test-normal-sw-clean-broken.jsonl', 'r', encoding='utf-8') as swfile:
        with open(f'/home/obehre/gsm-eval/dataset_creation/dataset_cleanup/test_normal/test-normal-sw-clean.jsonl', 'w', encoding='utf-8') as outfile:
            for wrong_line, right_line in zip(swfile, enfile, strict=True):
                # Parse the JSON content
                wrong_data = json.loads(wrong_line)
                right_data = json.loads(right_line) 

                # Extract from right_data["answer"] the part behind the last "\n"
                answer = right_data["answer"]
                cleaned_answer = answer.split('\n')[-1]
                print(cleaned_answer)
                
                # Substitute everything behind the last "\n" from wrong_data["answer"] with cleaned_answer
                wrong_answer_prefix = "\n".join(wrong_data["answer"].split('\n')[:-1])
                new_answer = f"{wrong_answer_prefix}\n{cleaned_answer}"
                print(new_answer)

                wrong_data["answer"] = new_answer

                # Write the decoded JSON data to the new file
                json.dump(wrong_data, outfile, ensure_ascii=False)
                outfile.write('\n')  # Add a newline after each JSON object