from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저 로드
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPU에 자동으로 할당
    torch_dtype=torch.bfloat16,  # bfloat16 데이터 타입 사용
    low_cpu_mem_usage=True  # 메모리 최적화
)
model = model.to(device)  # 모델을 GPU로 이동
# 데이터셋 로드
dataset = load_dataset("json", data_files="genAI\converted_jeonla.json")  # JSON 파일 경로

# Train/Validation 분할
train_test_split = dataset["train"].train_test_split(test_size=0.2)
tokenized_dataset = {
    "train": train_test_split["train"],
    "validation": train_test_split["test"]  # 'test'를 'validation'으로 변경
}
print(tokenized_dataset)  # 데이터셋 구조 확인

# 데이터 전처리 함수
def preprocess_data(example):
    # input과 output을 토크나이징
    input_encodings = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
    output_encodings = tokenizer(example["output"], truncation=True, padding="max_length", max_length=512)
    # 라벨 추가 (output 토큰화 결과를 라벨로 사용)
    labels = output_encodings["input_ids"]
    # return 값 구성
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": labels
    }

# 데이터셋 전처리
tokenized_dataset = {k: v.map(preprocess_data, batched=True) for k, v in tokenized_dataset.items()}

# 학습 설정 args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # GPU 메모리에 따라 조정
    per_device_eval_batch_size=1,   # GPU 메모리에 따라 조정
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # 16-bit 부동소수점 사용
    gradient_checkpointing=True,  # 메모리 절약을 위한 그라디언트 체크포인팅
    optim="adamw_torch",  # 최적화 알고리즘 설정
    gradient_accumulation_steps=8,  # 그래디언트 누적 단계
    dataloader_num_workers=4,  # 데이터 로더 워커 수
    report_to="none"  # 로그 리포팅 설정
)
# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)
# 모델 학습
trainer.train()
# 모델 저장
model.save_pretrained("./fine_tuned_gemma2-2b")