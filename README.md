# MedQA-Summarizer

A medical question summarization project using state-of-the-art NLP techniques.

## Project Overview

This project implements a T5-based model for summarizing medical questions. It can identify the more informative question from a pair of related medical queries.

## Key Features

- Fine-tuned T5 model for medical question summarization
- Competitive performance against BART and LED models
- Simple inference script for easy interaction with the model

## Results

Our fine-tuned T5 model achieved the following ROUGE scores on the test set:
- ROUGE-1: 0.6590
- ROUGE-2: 0.5517
- ROUGE-L: 0.6219

## Future Work

- Experiment with fine-tuning BART or LED models on our dataset
- Investigate techniques to improve ROUGE-2 scores
- Conduct qualitative analysis of model predictions
- Explore ensemble methods combining multiple models

## Getting Started

pip install -r requirements.txt

## Usage

run the following in order: data_analysis.py > data_preprocessing.py > train_model.py > evaluate_model.py > inference.py > model.comparision.py

- run `python3 app.py`
- open 127.0.0.1:5000 to try the inputs

## Contributors

feel free to contribute to this project, explore fine tuning other models 

