import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

def load_data(file_path):
    return torch.load(file_path)

def create_dataloaders(train_data, val_data, batch_size=4):
    train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
    val_dataset = TensorDataset(val_data['input_ids'], val_data['attention_mask'], val_data['labels'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    return {
        'rouge1': np.mean([score['rouge1'].fmeasure for score in scores]),
        'rouge2': np.mean([score['rouge2'].fmeasure for score in scores]),
        'rougeL': np.mean([score['rougeL'].fmeasure for score in scores]),
    }

def train(model, train_dataloader, val_dataloader, tokenizer, device, num_epochs=10, gradient_accumulation_steps=4):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            
            total_loss += loss.item()
            
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_train_loss = total_loss * gradient_accumulation_steps / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_inputs = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, num_beams=4, early_stopping=True)
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)
                all_inputs.extend(decoded_inputs)
        
        avg_val_loss = val_loss / len(val_dataloader)
        rouge_scores = compute_rouge(all_preds, all_labels)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"ROUGE scores: {rouge_scores}")
        
        # Print some examples
        print("\nExample predictions:")
        for i in range(min(3, len(all_inputs))):
            print(f"Input: {all_inputs[i]}")
            print(f"Prediction: {all_preds[i]}")
            print(f"Actual: {all_labels[i]}")
            print()
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_data = load_data('data/train_data.pt')
    val_data = load_data('data/val_data.pt')

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(train_data, val_data, batch_size=4)

    # Initialize model and tokenizer
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Train the model
    trained_model = train(model, train_dataloader, val_dataloader, tokenizer, device)

    # Save the trained model
    trained_model.save_pretrained('models/trained_summarizer')
    tokenizer.save_pretrained('models/trained_summarizer')
    print("Model saved successfully.")

if __name__ == "__main__":
    main()