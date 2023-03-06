from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

#text = "def greet(user): print(f'hello <extra_id_0>!')"
text = "#define a function that says hello world\n"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length=8)
#print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print(tokenizer.decode(generated_ids[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))

