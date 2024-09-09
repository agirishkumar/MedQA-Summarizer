import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer, LEDForConditionalGeneration, LEDTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from rouge_score import rouge_scorer
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['input'] = df.apply(lambda row: f"summarize: {row['question_1']} [SEP] {row['question_2']}", axis=1)
    df['target'] = df.apply(lambda row: row['question_2'] if row['label'] == 1 else row['question_1'], axis=1)
    return df[['input', 'target']]

def tokenize_data(df, tokenizer, max_length=512):
    inputs = tokenizer(df['input'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    return {
        'rouge1': sum(score['rouge1'].fmeasure for score in scores) / len(scores),
        'rouge2': sum(score['rouge2'].fmeasure for score in scores) / len(scores),
        'rougeL': sum(score['rougeL'].fmeasure for score in scores) / len(scores),
    }

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, num_beams=4, early_stopping=True)
            decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_preds.extend(decoded_preds)
    
    return all_preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    df = load_data('data/processed_data.csv')
    df = preprocess_data(df)
    
    # Split data into train and test
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Define models to compare
    models = {
        "Our Fine-tuned T5": ("models/trained_summarizer", T5ForConditionalGeneration, T5Tokenizer),
        "Pre-trained T5": ("t5-small", T5ForConditionalGeneration, T5Tokenizer),
        "BART": ("facebook/bart-base", BartForConditionalGeneration, BartTokenizer),
        "LED": ("allenai/led-base-16384", LEDForConditionalGeneration, LEDTokenizer),
    }

    results = {}

    for model_name, (model_path, model_class, tokenizer_class) in models.items():
        print(f"\nEvaluating {model_name}")
        model = model_class.from_pretrained(model_path).to(device)
        tokenizer = tokenizer_class.from_pretrained(model_path)

        # Tokenize test data
        test_inputs = tokenize_data(test_df, tokenizer)
        test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'])
        test_dataloader = DataLoader(test_dataset, batch_size=8)

        # Evaluate
        predictions = evaluate(model, test_dataloader, tokenizer, device)
        
        # Compute ROUGE scores
        rouge_scores = compute_rouge(predictions, test_df['target'].tolist())
        results[model_name] = rouge_scores

    # Print results
    print("\nResults:")
    for model_name, scores in results.items():
        print(f"\n{model_name}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

    

if __name__ == "__main__":
    main()