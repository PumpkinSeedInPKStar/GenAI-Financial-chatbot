from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Flask 앱 초기화
app = Flask(__name__)

# 모델 및 토크나이저 로드
model_name = "euneeei/team_gemma2_finetuned"  # Hugging Face Model Hub에 업로드한 이름
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("input_text", "")

    # 입력 텍스트 토크나이징
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # 모델 추론
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    # 출력 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 응답 반환
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
