import gc
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, GenerationConfig
from peft import PeftConfig, PeftModel
from huggingface_hub.hf_api import HfFolder

HfFolder.save_token('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')

token = "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"

data = pd.read_csv('TestNERDataset.csv', encoding='cp1252')

dataset = []
i = 0

inputs = data.get('text')
resp = data.get('chemicals')
inst = "Identify the chemicals in this text."

for i in range(len(inputs)):
    dataset.append(
        {
            'instruction': inst,
            'input': inputs[i],
            'output': resp[i],
            'source': "### Task: {instruction}\n### Input: {inp}\n### Answer: ".format(instruction=inst.strip(),
                                                                                       inp=inputs[i].strip()),
            'raw_entities': 'chemical',
            'id': f"{i}"
        }
    )
    i += 1

# model_name = "chrohi/llama-chat-ft-7b"
model_name = "meta-llama/Llama-2-7b-chat-hf"

generation_config = GenerationConfig.from_pretrained(model_name, token=token)

peft_config = PeftConfig.from_pretrained(model_name, token=token)
base_model_name = peft_config.base_model_name_or_path

models = {'llama': AutoModelForCausalLM, 't5': T5ForConditionalGeneration, 'mistral': AutoModelForCausalLM}

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    device_map='auto',
    token=token
)

model = PeftModel.from_pretrained(model, model_name, token=token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

model.eval()
model = torch.compile(model)

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

extracted_list = []
target_list = []
instruction_ids = []
sources = []


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        try:
            t = iterable[ndx:min(ndx + n, l)]
        except Exception:
            print("Error")
        yield t


# exit(0)
test_dataset = dataset
for instruction in tqdm(test_dataset):
    target_list.append(instruction['raw_entities'])
    instruction_ids.append(instruction['id'])
    sources.append(instruction['source'])

target_list = list(batch(target_list, n=1))
instruction_ids = list(batch(instruction_ids, n=1))
sources = list(batch(sources, n=1))

with wandb.init(project="Instruction NER") as run:
    for source in tqdm(sources):
        input_ids = tokenizer(source, return_tensors="pt", padding=True)["input_ids"].cuda()
        with torch.no_grad():
            # torch.cuda.empty_cache()
            # gc.collect()
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
        for s in generation_output.sequences:
            string_output = tokenizer.decode(s, skip_special_tokens=True)
            extracted_list.append(string_output)
            print("Final -> ", string_output)

pd.DataFrame({
    'id': np.concatenate(instruction_ids),
    'extracted': extracted_list,
    'target': np.concatenate(target_list)
}).to_json("finetuned_llama_chat_7b_testing_result.json")
