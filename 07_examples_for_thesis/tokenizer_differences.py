from transformers import AutoTokenizer

# Tokenizes the same sentence with two different tokenizers.
# Not used for the thesis. 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

example_sentence = "James beschließt, drei Mal pro Woche 3 Sprints zu laufen. Er läuft 60 Meter pro Sprint. Wie viele Meter läuft er pro Woche?"

encoded = tokenizer(example_sentence)

decoded = []
ids = encoded["input_ids"]
encoded = tokenizer(example_sentence, return_tensors="pt")
for i, token in enumerate(encoded["input_ids"][0]):
    decoded.append(tokenizer.decode(token, skip_special_tokens=True))
print(decoded)
print(ids)
print(len(ids))


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
encoded = tokenizer(example_sentence)

# Print encoded result in a nice list format
print("Tokenized output:")
decoded = []
ids = encoded["input_ids"]
encoded = tokenizer(example_sentence, return_tensors="pt")
for i, token in enumerate(encoded["input_ids"][0]):
    decoded.append(tokenizer.decode(token, skip_special_tokens=True))
print(decoded)
print(ids)
print(len(ids))

print(tokenizer.decode(ids))


