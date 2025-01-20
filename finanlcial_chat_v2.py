import os
from dataclasses import dataclass, field
from typing import Optional
import re

import torch
import sys
import tyro
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TextStreamer,
    logging as hf_logging,
)
import logging
from trl import SFTTrainer, SFTConfig

from trl.trainer import ConstantLengthDataset


base_model_id = "google/gemma-2b-it" # 다른 모델을 쓰고 싶다면, 이걸 바꿔주면 됩니당.
device_map="auto"
torch_dtype = torch.bfloat16
dataset_name = "C:/Users/user/Desktop/Sungshin/4-2/GENAI/genAI/converted_jeonla.jsonl"
seq_length = 512 # 512

full_dataset = Dataset.from_json(path_or_paths=dataset_name)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id
)
tokenizer.padding_side = "right"


lora_config = LoraConfig(
    r=6,
    lora_alpha = 8,
    lora_dropout = 0.1,
    target_modules=["q_proj", "k_proj","v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",  # GPU에 자동으로 할당
    torch_dtype=torch.float16,  # FP16 사용
    low_cpu_mem_usage=True,  # 메모리 최적화
    offload_folder="./offload"  # CPU로 일부 데이터 이동
)


model.config.use_cache = False

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id 
tokenizer.padding_side = "right"
if model.config.pad_token_id != tokenizer.pad_token_id:
    model.config.pad_token_id = tokenizer.pad_token_id


def chars_token_ratio(dataset, tokenizer, prepare_sample_text, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

# 복잡한 토큰화 설정을 단순화하며, 일관된 형식으로 모델에 입력을 제공
def function_prepare_sample_text(tokenizer, for_train=True):
    def _prepare_sample_text(example):
        """Prepare the text from a sample of the dataset."""
        user_prompt="너는 사용자가 입력한 금융 관련 질문을 분석하는 에이전트이다. 질문으로부터 금융 상담을 해야 한다.\n### 답변: "
        messages = [ # 이는 모델마다 형식이 조금씩 다르기 때문에 이는 개발자가 알아서 수정해야 하는 것(수작업)
            # {"role": "system", "content": f"{system_prompt}"}, # 이는 어떤 모델에선 "system"이 있는데, 다른 모델에서는 "system"이 없음.
            {"role": "user", "content": f"{user_prompt}{example['input']}"},
        ]
        if for_train: # 테스트 할 때는 이 파트가 빠져있음
            messages.append({"role": "assistant", "content": f"{example['output']}"})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False if for_train else True)
        return text
    return _prepare_sample_text

def create_datasets(tokenizer, dataset, seq_length):
  # 데이터셋을 얻는 것.
  # 코잘(Causal) 언어 모델을 훈련하는 데 사용될 고정 길이 시퀀스 데이터셋을 구성

    prepare_sample_text = function_prepare_sample_text(tokenizer)

    chars_per_token = chars_token_ratio(dataset, tokenizer, prepare_sample_text)
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}"
    )

    cl_dataset = ConstantLengthDataset(
        tokenizer,
        dataset,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    return cl_dataset # 생성된 ConstantLengthDataset 객체를 반환

ds = create_datasets(tokenizer, full_dataset, seq_length)
it = iter(ds)
try:
    decoded_text = tokenizer.decode(next(it)['input_ids'])
    print(decoded_text)
except StopIteration:
    print("End of dataset reached. Resetting iterator.")
    it = iter(ds)
    decoded_text = tokenizer.decode(next(it)['input_ids'])
    print(decoded_text)

# 학습 설정
training_args = SFTConfig(
    output_dir="./results",  # 모델 저장 위치
    evaluation_strategy="no",  # 매 에포크 종료 시 평가
    save_strategy="epoch",  # 매 에포크 종료 시 체크포인트 저장
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # GPU 메모리 최적화
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # 작은 배치를 누적하여 학습
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # Mixed Precision 사용
    fp16_full_eval=False,
    save_total_limit=2,  # 체크포인트 최대 2개만 유지
    report_to="none"  # 로그 비활성화
)

# Trainer 객체 생성
trainer = SFTTrainer(
    model=model,
    # train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["validation"],
    train_dataset=ds,
    eval_dataset=None,
    max_seq_length=512,
    args=training_args,
    peft_config=lora_config
    # formatting_func=generate_prompt,
)

torch.cuda.empty_cache()

# 모델 학습
trainer.train()
print(torch.cuda.memory_summary(device=torch.device("cuda:0"), abbreviated=True))

# 모델 저장
model.save_pretrained("./fine_tuned_gemma2_v3")
tokenizer.save_pretrained("./fine_tuned_gemma2_v3")