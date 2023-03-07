from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = """
You are a helpful assistant that generates code in a proprietary language called GPAL.
This is how to create views in GPAL:
views {
  view(“TRADE_VIEW”, TRADE) {
    joins {
      joining(INSTRUMENT) {
        on(TRADE.INSTRUMENT_ID to INSTRUMENT { INSTRUMENT_ID })
      }
    }
    fields {
      TRADE.allFields()
      INSTRUMENT.NAME withPrefix INSTRUMENT
      INSTRUMENT.CURRENCY_ID withAlias “CURRENCY”
    }
  }
}
create a view using GPAL that joins TRADE and ALLOCATION tables where the join is TRADE_ID and everything from a allocation table should be returned as well as a TRADE_ID from TRADE table    
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)