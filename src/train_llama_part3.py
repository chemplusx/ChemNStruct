import argparse
import json
import os
import pathlib

import numpy as np
import torch
from enum import auto, Enum
from torch.utils.data import Dataset
import wandb
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer,
                          DataCollatorForSeq2Seq, LlamaTokenizer,
                          DataCollatorForTokenClassification, EvalPrediction,
                          T5ForConditionalGeneration, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
from transformers.trainer_pt_utils import LabelSmoother
# from trainer_utils import PREFIX_CHECKPOINT_DIR
import random
from utils.train_utils import set_random_seed, fix_tokenizer, fix_model
from utils.metric import calculate_metrics, extract_classes
from schemas.ChemStruct import ChemStruct, InstructDataset
from huggingface_hub.hf_api import HfFolder
from conversation import Conversation, get_conv_template

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

local_rank = None
HfFolder.save_token('<token>')

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    NEW_LINE = auto()

class BaseAdapter:
    """The base and the default model adapter."""

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


class IEasQAAdapter(BaseAdapter):
    """The model adapter for FreedomIntelligence/phoenix-inst-chat-7b"""

    def match(self, model_path: str):
        return "ie_as_qa" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ie_as_qa")

model_adapters: list[BaseAdapter] = []

def get_model_adapter(model_path: str) -> BaseAdapter:
    """Get a model adapter for a model_path."""
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter
    raise ValueError(f"No valid model adapter for {model_path}")

def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())

register_model_adapter(IEasQAAdapter)

conv_templates: dict[str, Conversation] = {}

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{name} has been registered."
    conv_templates[template.name] = template

register_conv_template(
    Conversation(
        name="ie_as_qa",
        system="A virtual assistant answers questions from a user based on the provided paragraph.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

def get_conversation_template(model_path: str) -> Conversation:
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

def tokenize(tokenizer, prompt, add_special_tokens=False):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )
    return result.input_ids

def get_model_adapter(model_path: str) -> BaseAdapter:
    """Get a model adapter for a model_path."""
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter
    raise ValueError(f"No valid model adapter for {model_path}")

def get_conversation_template(model_path: str) -> Conversation:
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

def rank0_print(*args):
    if local_rank == 0 or local_rank == -1:
        print(*args)


# def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
#     # """Collects the state dict and dump to disk."""
#     # state_dict = trainer.model.state_dict()
#     # if trainer.args.should_save:
#     #     cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#     #     del state_dict
#     #     trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
#     model = trainer.model
#     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#     with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
#         cpu_state_dict = model.state_dict()
#         if trainer.args.should_save:
#             trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess2(
        sources,
        tokenizer: PreTrainedTokenizer,
) -> dict:
    conv = get_conversation_template("ie_as_qa")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    input_ids, labels = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv_input_ids, conv_targets = [], []

        # Randomly shuffle the queries, since the answer should be order-insensitive to queries.
        assert len(source) % 2 == 0
        idxs = list(range(len(source) // 2))[1:]
        random.shuffle(idxs)
        idxs = [0] + idxs
        new_source = []
        for rand_i in idxs:
            new_source.append(source[2 * rand_i])
            new_source.append(source[2 * rand_i + 1])
        source = new_source

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if j == 0:
                # first input consists of system message and user input
                message = conv.system + conv.sep + role + ": " + sentence["value"]
                _input_ids = tokenize(tokenizer, message, True)
                conv_input_ids += _input_ids
                conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
            else:
                if j % 2 == 0:
                    # user input
                    message = role + ": " + sentence["value"]
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
                else:
                    # assistant output
                    message = role + ": "
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
                    message = sentence["value"] + conv.sep2
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += _input_ids

        assert len(conv_input_ids) == len(conv_targets), source

        if len(conv_input_ids) < tokenizer.model_max_length:
            pad_length = tokenizer.model_max_length - len(conv_input_ids)
            conv_input_ids += [tokenizer.pad_token_id] * pad_length
            conv_targets += [IGNORE_TOKEN_ID] * pad_length

        input_ids.append(conv_input_ids[:tokenizer.model_max_length])
        labels.append(conv_targets[:tokenizer.model_max_length])

        if False:
            cur_user_ids, cur_gpt_ids = [], []
            for i, t in zip(conv_input_ids, conv_targets):
                if t == IGNORE_TOKEN_ID:
                    if i != tokenizer.pad_token_id:
                        cur_user_ids.append(i)
                    if len(cur_gpt_ids):
                        rank0_print(f"OUTPUT: '{tokenizer.decode(cur_gpt_ids)}'")
                        cur_gpt_ids = []
                else:
                    assert i == t, breakpoint()
                    cur_gpt_ids.append(t)
                    if len(cur_user_ids):
                        rank0_print(f"INPUT: '{tokenizer.decode(cur_user_ids)}'")
                        cur_user_ids = []
            assert len(cur_user_ids) == 0, breakpoint()
            if len(cur_gpt_ids):
                rank0_print(f"OUTPUT: '{tokenizer.decode(cur_gpt_ids)}'")
            breakpoint()

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess2(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess2([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: PreTrainedTokenizer, data_args
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args["lazy_preprocess"] else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args["data_path"], "r"))

    # Split train/test
    np.random.seed(3)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


tt = {
"run_name": "universalner",
"bf16": "True",
"output_dir": "./saved_models/universalner",
"dataloader_num_workers": 8,
"num_train_epochs": 1,
"per_device_train_batch_size": 8,
"per_device_eval_batch_size": 8,
"gradient_accumulation_steps":8,
"evaluation_strategy": "steps",
"eval_steps": 8,
"save_strategy": "steps",
"save_steps":8,
"save_total_limit": 2,
"learning_rate":2e-5,
"weight_decay": 0.,
"warmup_ratio": 0.04,
"lr_scheduler_type": "cosine",
"logging_steps": 1,
# "fsdp": "full_shard auto_wrap",
# "fsdp_config": "/mnt/d/workspace/universal-ner/src/train/fsdp_config.json",
"tf32": "True",

"gradient_checkpointing": "True",

}

ff = {
"model_name_or_path": "meta-llama/Llama-2-7b-hf",
"model_max_length": 1024,
"lazy_preprocess": "Tr",
"data_path": "./output_ner_convo_12.json",
}

def train():
    global local_rank

    # parser = HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments)
    # )
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # local_rank = training_args.local_rank
    training_args = TrainingArguments(
        **tt
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     ff["model_name_or_path"],
    # )

    model = AutoModelForCausalLM.from_pretrained(
        ff["model_name_or_path"],
        load_in_8bit=True,
        device_map='auto',
        use_flash_attention_2=False
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        ff["model_name_or_path"],
        model_max_length=ff["model_max_length"],
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = 0
    # model = fix_model(model, tokenizer, use_resize=False)
    # model = prepare_model_for_kbit_training(model)
    # lora_config = {
    #     "r": 8,
    #     "lora_alpha": 16,
    #     "lora_dropout": 0.05,
    #     "bias": "none",
    #     "target_modules": ["q_proj", "v_proj"],
    #     "task_type": "CAUSAL_LM"
    # }
    # peft_config = LoraConfig(**lora_config)
    # model = get_peft_model(model, peft_config)

    model.config.use_cache = False

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=ff)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.push_to_hub("ChemPlusX/llama2-7b-ner-type2", use_auth_token=True)
    tokenizer.push_to_hub("ChemPlusX/llama2-7b-ner-type2", use_auth_token=True)
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

train()
