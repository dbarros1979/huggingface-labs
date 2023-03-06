from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SalesForce/codegen-16B-mono")
model = AutoModelForCausalLM.from_pretrained("SalesForce/codegen-16B-mono", revision="sharded", device_map="auto", offload_folder="/offload_folder")

inputs = tokenizer("#define a function that says hello world\n", return_tensors="pt").to(0)
gen_ids = model.generate(**inputs, max_length=128)

#Specify the truncation pattern with regex
print(tokenizer.decode(gen_ids[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))

# Output
'''
#define a function that says hello world
def hello_world():
    print("Hello World")
'''
