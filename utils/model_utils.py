# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
# from peft import LoraConfig, PeftConfig
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from .configs import DataArguments, ModelArguments
from .data import DEFAULT_CHAT_TEMPLATE
# from peft import PeftModel, PeftConfig


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


# def get_quantization_config(model_args: ModelArguments) -> BitsAndBytesConfig | None:
#     if model_args.load_in_4bit:
#         compute_dtype = torch.float16
#         if model_args.torch_dtype not in {"auto", None}:
#             compute_dtype = getattr(torch, model_args.torch_dtype)

#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=compute_dtype,
#             bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
#             bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
#         )
#     elif model_args.load_in_8bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#         )
#     else:
#         quantization_config = None

#     return quantization_config


def get_tokenizer(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    # if data_args.chat_template is not None:
        # tokenizer.chat_template = data_args.chat_template
    # elif tokenizer.chat_template is None and tokenizer.default_chat_template is None:
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


# def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
#     if model_args.use_peft is False:
#         return None

#     peft_config = LoraConfig(
#         r=model_args.lora_r,
#         lora_alpha=model_args.lora_alpha,
#         lora_dropout=model_args.lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=model_args.lora_target_modules,
#         modules_to_save=model_args.lora_modules_to_save,
#     )

#     return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


# def get_checkpoint(training_args: SFTConfig | DPOConfig) -> Path | None:
#     last_checkpoint = None
#     if os.path.isdir(training_args.output_dir):
#         last_checkpoint = get_last_checkpoint(training_args.output_dir)
#     return last_checkpoint


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model(data_args, model_args, training_args, tokenizer, logger):
    
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                  attn_implementation=model_args.attn_implementation
                                                 )
    model.config.use_cache = False

    if model_args.use_peft:
        if model_args.peft_model_id:
            model = PeftModel.from_pretrained(model, model_args.peft_model_id)
            ## If still need to fine-tune
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
        else:
            config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_rank * 2,
                target_modules=["down_proj"], # model specific, doesn't work with microsoft/phi-2, for microsoft/phi-2 use ["q_proj", "k_proj", "v_proj", "dense"]
                # target_modules=["q_proj", "k_proj", "v_proj", "dense"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
        print_trainable_parameters(model)
    return model
