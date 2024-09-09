from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__, static_folder='static')
CORS(app)  # This will enable CORS for all routes

# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = T5ForConditionalGeneration.from_pretrained('models/trained_summarizer').to(device)
tokenizer = T5Tokenizer.from_pretrained('models/trained_summarizer')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    if 'question1' not in data or 'question2' not in data:
        return jsonify({'error': 'Both question1 and question2 are required'}), 400

    question1 = data['question1']
    question2 = data['question2']

    input_text = f"summarize: {question1} [SEP] {question2}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)

    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({
        'summary': summary,
        'original_questions': {
            'question1': question1,
            'question2': question2
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)