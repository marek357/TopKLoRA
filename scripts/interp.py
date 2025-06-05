from transformers import AutoTokenizer, AutoModelForCausalLM
import os

adapters = set()

for root, dir, files in os.walk('../adapters/sft'):
    for file in files:
        # don't understand the magic happening here
        adapters.add(root)

for adapter in adapters:
    try:
        model = AutoModelForCausalLM.from_pretrained(adapter)
        print(model)
    except RuntimeError:
        pass
