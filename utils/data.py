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
from typing import List, Literal, Optional

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
import json
from .configs import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "dpo"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )

    elif task == "tpo":
        if all(k in example.keys() for k in ("chosen", "rejected", "reference")):
            if example["prompt"] is None:
                    raise ValueError("Prompt is empty")
            # For TPO, the inputs are four of (prompt, reference, chosen, rejected), where `reference` is gold response and `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = [{"content": example['prompt'], "role": "user"}]
            # Prepend a system message if the first message is not a system message
            # if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            reference_messages = example["reference"]

            chosen_messages =  [{'content':example['chosen'], 'role': 'assistant'}]
            rejected_messages = [{'content':example['rejected'], 'role': 'assistant'}]
            reference_messages = [{'content':example['reference'], 'role': 'assistant'}]

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_reference"] = tokenizer.apply_chat_template(reference_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'tpo']}"
        )
    return example



def get_datasets(dataset_name_or_path:str = None, local_dataset: bool = False) -> Dataset:
    dataset = None
    if local_dataset:
        try:
            dataset = load_dataset('json', data_files=dataset_name_or_path)
        except:
            dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    print(dataset)    
    return dataset


