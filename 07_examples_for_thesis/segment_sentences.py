from transformers import AutoTokenizer

# Tokenization of a sentences for an example in the foundation.

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

example_sentence = "Math word problems can be a backbreaker!"

encoded = tokenizer(example_sentence)

decoded = []
ids = encoded["input_ids"]
encoded = tokenizer(example_sentence, return_tensors="pt")
for i, token in enumerate(encoded["input_ids"][0]):
    decoded.append(tokenizer.decode(token, skip_special_tokens=True))
print(decoded)
print(ids)


# ['Math', '␣word', '␣problems', '␣can', '␣be', '␣a', '␣back', 'breaker', '!']
# [8991, 3492, 5435, 649, 387, 264, 1203, 65221, 0]