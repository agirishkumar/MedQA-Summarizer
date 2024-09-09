import os
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def analyze_data(df):
    print(f"Total samples: {len(df)}")
    print("\nColumn names:")
    print(df.columns)
    
    print("\nSample data:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Analyze question lengths
    df['question_1_length'] = df['question_1'].str.len()
    df['question_2_length'] = df['question_2'].str.len()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df['question_1_length'].hist(bins=50)
    plt.title('Distribution of Question 1 Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df['question_2_length'].hist(bins=50)
    plt.title('Distribution of Question 2 Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data/length_distributions.png')
    print("\nLength distribution plot saved as 'length_distributions.png'")

    # Analyze label distribution
    label_counts = df['label'].value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    
    plt.figure(figsize=(8, 6))
    label_counts.plot(kind='bar')
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig('data/label_distribution.png')
    print("\nLabel distribution plot saved as 'label_distribution.png'")

def main():
    # Load the dataset
    dataset = load_dataset("medical_questions_pairs")
    df = pd.DataFrame(dataset['train'])

    # Analyze the data
    analyze_data(df)

    # Tokenize and analyze token lengths
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    def tokenize_and_analyze(text):
        return len(tokenizer.encode(text))

    df['question_1_tokens'] = df['question_1'].apply(tokenize_and_analyze)
    df['question_2_tokens'] = df['question_2'].apply(tokenize_and_analyze)

    print("\nToken statistics:")
    print(df[['question_1_tokens', 'question_2_tokens']].describe())

    # Save processed data
    df.to_csv('data/processed_data.csv', index=False)
    print("\nProcessed data saved as 'processed_data.csv'")

if __name__ == "__main__":
    main()