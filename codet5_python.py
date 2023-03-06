from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py")

text = "#define a function that says hello world in Kotlin\n"
input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
