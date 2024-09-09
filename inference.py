import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer, device

def generate_summary(model, tokenizer, device, question1, question2):
    input_text = f"summarize: {question1} [SEP] {question2}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def main():
    model_path = 'models/trained_summarizer'
    model, tokenizer, device = load_model(model_path)
    
    print("Welcome to the Medical Question Summarizer!")
    print("Enter two related medical questions, and the model will select the more informative one.")
    print("Enter 'quit' at any time to exit.")
    
    while True:
        question1 = input("\nEnter the first question: ")
        if question1.lower() == 'quit':
            break
        
        question2 = input("Enter the second question: ")
        if question2.lower() == 'quit':
            break
        
        summary = generate_summary(model, tokenizer, device, question1, question2)
        
        print("\nThe model selected the following as the more informative question:")
        print(summary)

if __name__ == "__main__":
    main()