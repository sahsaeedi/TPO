<div align="center">

# TPO: Triple Preference Optimization
</div>


This repository contains the code and released models for our paper [Triple Preference Optimization: Achieving Better Alignment in a Single Step Optimization](https://arxiv.org/abs/2405.16681). We propose a novel preference optimization to enhance the instruction-following and reasoning capabilities of large language models in a single step, beginning with the pre-trained/instruction-tuned model. TPO outperforms DPO and its latest variants across MixEval-Hard, MT-Bench, and Arena-Hard, representing instruction following benchmarks, and GSm8K, MMLU, and MMLU-Pro represent the reasoning benchmarks under various settings. Please find all the released model checkpoints at [this link](https://huggingface.co/tpo-alignment). 

<img src="img/tpo_overview.png" width="1000px"></img>

<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>
<a href="https://arxiv.org/abs/2405.16681" alt="TPO paper"><img src="https://img.shields.io/badge/TPO-Paper-D9AB42" /></a>
<a href="https://www.asu.edu/" alt="jhu"><img src="https://img.shields.io/badge/Arizona_State_University-BEC23F" /></a>
<a href="https://twitter.com/sahsaeedi"><img src="https://img.shields.io/twitter/follow/sahsaeedi?style=social&logo=twitter" alt="follow on Twitter"></a>
<a href="https://twitter.com/ver_shivanshu"><img src="https://img.shields.io/twitter/follow/ver_shivanshu?style=social&logo=twitter" alt="follow on Twitter"></a>
<a href="https://twitter.com/krasul"><img src="https://img.shields.io/twitter/follow/krasul?style=social&logo=twitter" alt="follow on Twitter"></a>


## Contents

- [Install Requirements](#install-requirments-)
- [Training Script](#training-script-)
- [Running TPO](#running-tpo-)
- [Data Information](#training-with-tpo-)
- [Citations](#citations)


## Install Requirements
### Environment Setup
This is a quick tutorial to set up and train a model with the TPO method.

**Create and activate a Conda environment**:
```bash
conda create --prefix tpo python=3.10 
conda activate tpo
```
**Install PyTorch with CUDA support (for Nvidia GPUs)**:
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```
**Install additional requirements**:
You can then install the remaining package dependencies of [alignment-handbook](https://github.com/huggingface/alignment-handbook) as follows:

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```
To use our code, you need to downgrade the versions of the `trl` and `transformers` packages. Run the following command:
```bash
pip install -r requirements.txt
```

## Training Script
We provide a shell script (`run_tpo.sh`) for training a model with TPO. The default configuration is optimized for 4×A100 GPUs. Depending on your computing environment, you may need to adjust `num_processes` and `per_device_train_batch_size`.

```bash
#!/bin/bash

OUTPUT_DIR="./tpo"
DATASET_NAME_OR_PATH=""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
     run_tpo.py  \
    --model_name_or_path meta-llama/meta-Llama-3-8B \
    --tokenizer_name meta-llama/meta-Llama-3-8B  \
    --is_three_preference true \
    --beta 0.01  \
    --tpo_alpha 1  \
    --do_train  \
    --bf16   \
    --attn_implementation flash_attention_2 \
    --learning_rate 5.0e-7 \
    --gradient_accumulation_steps 1  \
    --lr_scheduler_type cosine  \
    --optim adamw_torch  \
    --warmup_ratio 0.1   \
    --save_steps 100  \
    --log_level info   \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1  \
    --evaluation_strategy steps   \
    --save_total_limit 1  \
    --logging_strategy steps \
    --logging_steps 10   \
    --output_dir $OUTPUT_DIR  \
    --num_train_epochs 1  \
    --max_length 1024   \
    --max_prompt_length 512 \
    --seed 42  \
    --overwrite_output_dir \
    --report_to none \
    --local_dataset \
    --dataset_name_or_path $DATASET_NAME_OR_PATH

```

To use `TPO-L` you just need to add the following setting to the above scritp.

```
    --tpo_l_gamma 5.4  \
    --loss_type tpo-l
```


## Running TPO
### Hyperparameter tuning
We used the following hyperparameters for training the released models.
| Training Size | Model                 | Method     | α       | β     | γ       | Learning Rate |
|--------------|-----------------------|-----------|--------|------|---------|---------------|
| 5K           | Llama-3-Base           | TPO/TPO-L | 1      | 0.01 | --/0.5  | 5e-7          |
| 5K           | Mistral-v0.3-Base      | TPO/TPO-L | 1/0.05 | 0.01/2 | --/1.6  | 5e-7          |
| 10K          | Llama-3-Base           | TPO/TPO-L | 1      | 0.01 | --/3    | 5e-7          |
| 10K          | Mistral-v0.3-Base      | TPO/TPO-L | 1/0.05 | 0.01/2 | --/1.6  | 5e-7          |
| 20K          | Llama-3-Base           | TPO/TPO-L | 1      | 0.01 | --/1.5  | 5e-7          |
| 20K          | Mistral-v0.3-Base      | TPO/TPO-L | 1      | 0.01/2 | --/1.6  | 5e-7          |
| 40K          | Llama-3-Base           | TPO/TPO-L | 1      | 0.01 | --/10   | 5e-7          |
| 40K          | Mistral-v0.3-Base      | TPO/TPO-L | 1/0.05 | 0.01/2 | --/1.6  | 5e-7          |
| 60K          | Llama-3-Instruct       | TPO/TPO-L | 0.05   | 0.01/10 | --/3    | 1e-6          |
| 60K          | Mistral-v0.2-Instruct  | TPO/TPO-L | 0.05   | 0.01/2.5 | --/0.3  | 1e-6          |


For DPO, the best hyperparameters for each setting are as follows.
| Setting                  | β | Learning Rate |
|------------------------|------|---------------|
| Mistral-Base           | 0.01 | 5e-7      |
| Llama3-Base            | 0.01 | 5e-7      |

>**Note:** For the Instruct setting, we used the latest checkpoints released by the SimPO paper, which can be found in this [repository](https://huggingface.co/collections/princeton-nlp/simpo-66500741a5a066eb7d445889).
## Released Models

| Llama Models                         | Repository Link | Arena-Hard | GSM8K |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------|-------|
| Llama 3 Instruct 8B TPO 40k | [tpo-alignment/Llama-3-8B-TPO-40k](https://huggingface.co/tpo-alignment/Llama-3-8B-TPO-40k)  |      6.9      |    51.2   |
| Llama 3 Instruct 8B TPO-L 40k | [tpo-alignment/Llama-3-8B-TPO-L-40k](https://huggingface.co/tpo-alignment/Llama-3-8B-TPO-L-40k)  |     12.3       |     52.4  |
| Llama 3 Instruct 8B TPO y2   | [tpo-alignment/Instruct-Llama-3-8B-TPO-y2](https://huggingface.co/tpo-alignment/Instruct-Llama-3-8B-TPO-y2)  |    42.0        |    77.2   |
| Llama 3 Instruct 8B TPO y3   | [tpo-alignment/Instruct-Llama-3-8B-TPO-y3](https://huggingface.co/tpo-alignment/Instruct-Llama-3-8B-TPO-y3)  |   42.4         |    77.8   |
| Llama 3 Instruct 8B TPO y4  | [tpo-alignment/Instruct-Llama-3-8B-TPO-y4](https://huggingface.co/tpo-alignment/Instruct-Llama-3-8B-TPO-y4)  |     38.9       |     78.2  |
| Llama 3 Instruct 8B TPO-L y2   | [tpo-alignment/Instruct-Llama-3-8B-TPO-L-y2](https://huggingface.co/tpo-alignment/Instruct-Llama-3-8B-TPO-L-y2)  |      39.4      |  77.3     |
---
---


| Mistral Models                         | Repository Link | Arena-Hard | GSM8K |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------------|-------|
| Mistral 7B TPO 40k | [tpo-alignment/Mistral-7B-TPO-40k](https://huggingface.co/tpo-alignment/Mistral-7B-TPO-40k)  |       7.4     |     39.2  |
| Mistral 7B Instrcut TPO y2 v0.1 | [tpo-alignment/Mistral-Instruct-7B-TPO-y2-v0.1](https://huggingface.co/tpo-alignment/Mistral-Instruct-7B-TPO-y2-v0.1)  |        26.2    |     40.6  |
| Mistral 7B Instrcut TPO y2 v0.2   | [tpo-alignment/Mistral-Instruct-7B-TPO-y2-v0.2](https://huggingface.co/tpo-alignment/Mistral-Instruct-7B-TPO-y2-v0.2)  |        24.9    |    43.2   |
| Mistral 7B Instrcut TPO y3   | [tpo-alignment/Mistral-Instruct-7B-TPO-y3](https://huggingface.co/tpo-alignment/Mistral-Instruct-7B-TPO-y3)  |     26.3       |   42.5    |
| Mistral 7B Instrcut TPO y4 | [tpo-alignment/Mistral-Instruct-7B-TPO-y4](https://huggingface.co/tpo-alignment/Mistral-Instruct-7B-TPO-y4)  |     27.2       |   42.2    |
---

### Use our models for inference
Please refer to the [generate.py](generate.py) script for detailed instructions on loading the model with the appropriate chat template.

## Evaluation
We follow the official implementation for evaluation on MixEval-Hard, Arena-Hard, and MT-Bench, MMLU-Pro as follows:

* MixEval-Hard: Please refer to the [MixEval repo](https://github.com/JinjieNi/MixEval?tab=readme-ov-file) for evaluation.

* Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

* MMLU-Pro: Please refer to the [MMLU-Pro repo](https://github.com/TIGER-AI-Lab/MMLU-Pro) for evaluation.

## Dataset Information

To train
TPO, which requires three preferences, we created
a custom dataset from the <a href="https://huggingface.co/datasets/openbmb/UltraFeedback">**UltraFeedback**</a> dataset.
Here, the response with the highest score serves as
the reference response, the second-highest score as
the chosen response, and the lowest score as the
rejected response.

Finally, the dataset includes < $y_{\text{gold}}$, $y_w$, $y_l$ > where $y_{\text{gold}}$ represents reference response, $y_{w}$ represents chosen response and $y_l$ represents rejected response. 

The Data Format in JSON file must be:
```JSON
{
    "prompt": "PROMPT_SENTENCE",
    "reference": "REFERENCE_SENTENCE",
    "chosen": "CHOSEN_SENTENCE",
    "rejected": "REJECTED_SENTENCE",
}
``` 
## Questions?
If you have any questions related to the code or the paper, feel free to email Amir (ssaeidi1@asu.edu).

## Citations
```bibtex
@misc{saeidi2025triplepreferenceoptimizationachieving,
      title={Triple Preference Optimization: Achieving Better Alignment using a Single Step Optimization}, 
      author={Amir Saeidi and Shivanshu Verma and Aswin RRV and Kashif Rasul and Chitta Baral},
      year={2025},
      eprint={2405.16681},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.16681}, 
}
```

For more insights about various alignment methods, please check <a href="https://arxiv.org/abs/2404.14723"> paper</a>.
```bibtex
@article{saeidi2024insights,
  title={Insights into Alignment: Evaluating DPO and its Variants Across Multiple Tasks},
  author={Saeidi, Amir and Verma, Shivanshu and Baral, Chitta},
  journal={arXiv preprint arXiv:2404.14723},
  year={2024}
}
```
    
## Acknowledgements
We thank the Research Computing (RC) at Arizona State University (ASU) and [cr8dl.ai](https://www.cr8dl.ai/) for their generous support in providing computing resources. The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers. We also, thanks to <a href="https://github.com/huggingface"> Hugging Face</a>
 for their <a href="http://hf.co/docs/trl"> Transformer Reinforcement Learning (TRL) </a> library, which greatly assisted in our project. 

