from transformers import AutoTokenizer, T5ForConditionalGeneration

hf_model_name = "icatlab-uiuc/codet5-normal"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = T5ForConditionalGeneration.from_pretrained(hf_model_name)

text = """
    You are a helpful assistant that generates code in a proprietary language called GPAL.
    This is how to create views in GPAL:
    views {
      view('TRADE_VIEW', TRADE) {
        joins {
          joining(INSTRUMENT) {
            on(TRADE.INSTRUMENT_ID to INSTRUMENT { INSTRUMENT_ID })
          }
        }
        fields {
          TRADE.allFields()
          INSTRUMENT.NAME withPrefix INSTRUMENT
          INSTRUMENT.CURRENCY_ID withAlias 'CURRENCY'
        }
      }
    }
    #create a view using GPAL that joins TRADE and ALLOCATION tables where the join is TRADE_ID and everything from a allocation table should be returned as well as a TRADE_ID from TRADE table
"""
input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
