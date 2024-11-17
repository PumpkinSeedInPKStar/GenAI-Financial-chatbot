from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("Gemma2-base")
model = AutoModelForCausalLM.from_pretrained("Gemma2-base")

# 데이터셋 로드 
dataset = load_dataset(r'C:\jeonla.json')  # JSON 파일 경로 또는 데이터셋 이름

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
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# 학습 설정 args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
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
model.save_pretrained("./fine_tuned_gemma2")
