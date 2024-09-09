import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import torch

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Format input for T5: "summarize: [TEXT]"
    df['input'] = df.apply(lambda row: f"summarize: {row['question_1']} [SEP] {row['question_2']}", axis=1)
    
    # Use the more informative question as the target
    df['target'] = df.apply(lambda row: row['question_2'] if row['label'] == 1 else row['question_1'], axis=1)
    
    return df[['input', 'target']]

def tokenize_data(df, tokenizer, max_length=512):
    inputs = tokenizer(df['input'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    targets = tokenizer(df['target'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    
    return {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': targets.input_ids
    }

def main():
    # Load the processed data
    df = load_data('data/processed_data.csv')
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Split the data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Tokenize the data
    train_data = tokenize_data(train_df, tokenizer)
    val_data = tokenize_data(val_df, tokenizer)
    
    # Save the preprocessed and tokenized data
    torch.save(train_data, 'data/train_data.pt')
    torch.save(val_data, 'data/val_data.pt')
    
    print(f"Preprocessed data saved. Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Print a few examples
    print("\nExample data:")
    for i in range(min(3, len(df))):
        print(f"Input: {df['input'].iloc[i]}")
        print(f"Target: {df['target'].iloc[i]}")
        print()

if __name__ == "__main__":
    main()