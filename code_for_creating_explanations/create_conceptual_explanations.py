#This file takes the information manually logged used collect_conceptual_information.py and turns it into natural language explanations.
#e.g. {('mass', 'weight'): '1', ('mass', 'gravity'): '1', ('weight', 'gravity'): '1'} -> "In the context of this question, the concept of mass is directly related to the concept of weight, the concept of mass is directly related to the concept of gravity and the concept of weight is directly related to the concept of gravity."
#1 indicates the concepts are directly related. 0 indicates they are inversely related

import json
import os
import nltk
import ast
import string
#nltk.download('punkt')
from nltk import word_tokenize
import ast
import dataset_remove_extra_entries as cl #in the order dev, test, train
JSONDATASETSPATH = 'measuremental_numerical_duplicate_source_to_add_conceptual' #replace this with the path to the folder the dev test and train dataset files are in
DATASETFILES = ['finalnumerical_measuremental_dev.json', 'finalnumerical_measuremental_test.json', 'finalnumerical_measuremental_train.json'] #the dev test and train dataset file names
#original datasets
CONCEPTUALNOTESPATH = 'conceptual_data' #replace with the path to the folder the explanation keywords created manually using collect_conceptual_information.py are in
CONCEPTUALNOTES = ['conceptual_dev.txt','conceptual_test.txt', 'conceptual_train.txt'] #the names of the files containing the explanation keywords of the dev test and train datasets

def load_dataset(path, names_in_order_dev_test_train, format):
    datasets_dict = {}
    labels = ['dev', 'test', 'train']
    for i, name in enumerate(names_in_order_dev_test_train):
        file_path = os.path.join(path, name)
        f = open(file_path)
        if format == 'json':
            datasets_dict[labels[i]] = json.load(f)
        elif format == 'txt':
            datasets_dict[labels[i]] = f
    return datasets_dict


#takes the data about the relationships between concepts from the txt files, makes explanatory sentences based on them and adds them to the json data.
def format_data(conceptual_data, json_data, name):
    for i, line in enumerate(conceptual_data):
        index, question_and_choices, concepts, relationships = line.split('\t')
        index = int(index)
        assert index == i
        concepts = concepts.strip('[]\'').split(', \'')
        relationships = ast.literal_eval(relationships)
        single_comparisons = []
        if len(concepts) > 1:
            for key in relationships.keys():
                if relationships[key] == '0':
                    comparison = f'the concept of {key[0]} is inversely related to the concept of {key[1]}'
                elif relationships[key] == '1':
                    comparison = f'the concept of {key[0]} is directly related to the concept of {key[1]}'
                else:
                    print(key, relationships[key])
                    raise Exception
                single_comparisons.append(comparison)
            if len(single_comparisons) > 2:
                print(single_comparisons, question_and_choices)
                final_comparison = 'In the context of this question, '
                for c in single_comparisons[:-2]:
                    final_comparison += f'{c}, '
                final_comparison += f'{single_comparisons[-2]} and '
                final_comparison += f'{single_comparisons[-1]}.'
            elif len(single_comparisons) == 2:
                final_comparison = f'In the context of this question, {single_comparisons[0]} and {single_comparisons[1]}.'
            else:
                final_comparison = f'In the context of this question, {single_comparisons[0]}.'
            json_data[index]['conceptual_explanation'] = final_comparison
            output_path = os.path.join(CONCEPTUALNOTESPATH, f'numerical_measuremental_conceptual_{name}.json')
    with open(output_path, 'w') as out:
        json.dump(json_data, out)
            #add final_comparison to the json dataset
                
    
            
                

def main():
    datasets = load_dataset(JSONDATASETSPATH, DATASETFILES, 'json')
    explanations = load_dataset(CONCEPTUALNOTESPATH, CONCEPTUALNOTES, 'txt')
    dev_explanations = explanations['dev']
    test_explanations = explanations['test']
    train_explanations = explanations['train']
    dev_datasets = datasets['dev']
    test_datasets = datasets['test']
    train_datasets = datasets['train']
    format_data(train_explanations, train_datasets, 'train')
    format_data(dev_explanations, dev_datasets, 'dev')
    format_data(test_explanations, test_datasets, 'test')
   

main()
