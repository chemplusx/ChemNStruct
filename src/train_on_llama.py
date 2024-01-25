import argparse
import json
import os

import numpy as np
import torch
import wandb
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          DataCollatorForTokenClassification, EvalPrediction,
                          T5ForConditionalGeneration, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.utils.train_utils import set_random_seed, fix_tokenizer, fix_model
from src.utils.metric import calculate_metrics, extract_classes
from src.schemas.ChemStruct import ChemStruct, InstructDataset


# https://github.com/huggingface/peft/issues/96
class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


def train(
        train_instructions: list[ChemStruct],
        test_instructions: list[ChemStruct],
        model_type: str,
        use_flash_attention_2: bool,
        output_dir: str,
        seed: int,
        config_file: str,
        push_to_hub: bool,
        hf_name_postfix: str
):
    set_random_seed(seed)
    with open(config_file, "r") as r:
        config = json.load(r)

    lora_config = config.get("lora")
    model_name = config['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer)

    def compute_metrics(eval_prediction: EvalPrediction, tokenizer=tokenizer):
        print("Calculating metric: ")
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        labels = eval_prediction.label_ids

        extracted_entities = []
        target_entities = []
        for ind, pred in enumerate(predictions):
            non_masked_indices = (labels[ind] != -100)
            pred = tokenizer.decode(pred, skip_special_tokens=True)
            label = tokenizer.decode(labels[ind][non_masked_indices], skip_special_tokens=True)

            extracted_entities.append(extract_classes(pred, ENTITY_TYPES))
            target_entities.append(extract_classes(label, ENTITY_TYPES))

        print("Calculated metric: ", pred, label, extracted_entities, target_entities)
        return calculate_metrics(extracted_entities, target_entities, ENTITY_TYPES, return_only_f1=True)

    only_target_loss = config.get("only_target_loss", True)
    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]

    train_dataset = InstructDataset(
        train_instructions,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        model_type=model_type,
        only_target_loss=only_target_loss
    )

    val_dataset = InstructDataset(
        test_instructions,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        model_type=model_type,
        only_target_loss=only_target_loss
    )

    model_classes = {
        'llama': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        'mistral': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        't5': {
            'data_collator': DataCollatorForSeq2Seq,
            'model': T5ForConditionalGeneration
        }
    }

    data_collator = model_classes[model_type]['data_collator'](tokenizer, pad_to_multiple_of=8)

    load_in_8bit = bool(config.get("load_in_8bit", True))
    is_adapter = config['is_adapter']
    if load_in_8bit:
        if is_adapter:
            peft_config = PeftConfig.from_pretrained(model_name)
            model = model_classes[model_type]['model'].from_pretrained(
                peft_config.base_model_name_or_path,
                load_in_8bit=True,
                device_map='auto',
                use_flash_attention_2=use_flash_attention_2
            )
            model = fix_model(model, tokenizer, use_resize=False)
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        else:
            model = model_classes[model_type]['model'].from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map='auto',
                use_flash_attention_2=use_flash_attention_2
            )
            model = fix_model(model, tokenizer, use_resize=False)
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = fix_model(model, tokenizer, use_resize=False)

    # Default model generation params
    model.config.num_beams = 5
    max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
    model.config.max_length = max_tokens_count

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='wandb',
        ddp_find_unused_parameters=None,
        deepspeed=deepspeed_config,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[SavePeftModelCallback],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    with wandb.init(project="ChemNStruct NER") as run:
        model.print_trainable_parameters()
        trainer.train()
        if 'llama2' in config_file:
            model_type = 'llama2'
        if push_to_hub:
            model.push_to_hub(f"chrohi/{model_type}-{hf_name_postfix}", use_auth_token=True)
            tokenizer.push_to_hub(f"chrohi/{model_type}-{hf_name_postfix}", use_auth_token=True)


if __name__ == "__main__":

    from src.schemas.DataStruct import create_train_test_instruct_datasets, ENTITY_TYPES

    train_dataset, test_dataset = create_train_test_instruct_datasets(
        data_path="data/annotated_nlm.json",
        max_instances=-1,
        test_size=0.2,
        random_seed=42
    )

    model_type = "llama"
    use_flash_attention = False
    output_dir = "output"
    random_seed = 42
    config_file = "configs/llama2_7b_lora.json"
    push_to_hub = True
    hf_name_postfix = "NLMC-NER-FT"

    train(
        train_instructions=train_dataset,
        test_instructions=test_dataset,
        model_type=model_type,
        use_flash_attention_2=use_flash_attention,
        output_dir=output_dir,
        seed=random_seed,
        config_file=config_file,
        push_to_hub=push_to_hub,
        hf_name_postfix=hf_name_postfix
    )
