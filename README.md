# RnD-BERT
Finetunes BERT on part of SemEval 2024 task 7.1 QQA with explanations
1. The .py files inside code_for_creating_explanations create explanations for the datasets of semeval 2024 task 7, numeval, subtask 1, QQA task, making use of the txt files in manual_additions_to_explanations. The QQA dataset is available here: https://sites.google.com/view/numeval/data
2.  mathbert_code and baseline_bert_code carry out hyperparameters tuning and finetune mathBERT and baseline BERT, respectively, on the dataset, using BertForMultipleChoice. The two files contain identical functions, but one runs them on BERT and one on MathBERT. This was done to avoid running the wrong models by mistake.
3.  prediction_analysis.py calculates Cohen's Kappa on all combinations of the test set results.
