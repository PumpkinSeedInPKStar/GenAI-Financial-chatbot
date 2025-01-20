import json

# Load the input JSON file
input_file = "C:/Users/user/Desktop/Sungshin/4-2/GENAI/genAI/converted_jeonla.json"  # Your input file name
output_file = "converted_jeonla.jsonl"  # Desired output file name in JSONL format

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create the output JSONL format
with open(output_file, "w", encoding="utf-8") as f_out:
    for entry in data:
        # Create the new formatted JSON object
        jsonl_entry = {
            "input": entry["input"],
            "output": entry["output"]
        }
        # Write each dictionary as a new line in JSONL format
        f_out.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

# Indicate completion
print(f"Data successfully converted to {output_file}")

