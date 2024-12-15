#This code goes through the dev, test and train datasets, and prompts the user to manually enter the concepts found in each question+answers and the relationship between each possible pair of concepts (either 1, directly related, or 0, inversely related).
#The dataset contains groups of questions with the same formulation (and the same concepts) but using different numbers and measurements 
#e.g. "Mike and Sue decide to ride their bikes around the park. They follow the same path but Mike rides his bike at 16 miles/hour and Sue rides at 5 miles/hour. After ten minutes, who will have covered less distance?? ", "Option1": "Mike", "Option2": "Sue", "answer": "Option 2"
#"question": "Mike and Sue decide to ride their bikes around the park. They follow the same path but Mike rides his bike at 4.5 m/sec and Sue rides at 3.4 m/sec. After ten minutes, who will have covered more distance?? ", "Option1": "Mike", "Option2": "Sue", "answer": "Option 1"
#Only the first question is presented to the user. Subsequent questions are filled in automatically

#a small number of concepts appear very often. check_data() prints all the concepts except for them. It was used to gain understanding of the data and is not a necessary part the pipeline

import json
import os
import nltk
import ast
import string
#nltk.download('punkt')
from nltk import word_tokenize
import dataset_remove_extra_entries as cl #in the order dev, test, train
INPUTPATH = 'datasets' #replace with the path to the folder containing the datasets
INPUTFILES = ['QQA_dev.json', 'QQA_test.json', 'QQA_train.json'] #dataset file names
#original datasets
OUTPUTPATH = 'conceptual_data' #replace with the path to the folder the output files will be created in
CONCEPTUALFILES = ['conceptual_dev.txt','conceptual_test.txt', 'conceptual_train.txt'] #the names of the files to be created

def load_dataset(path, names_in_order_dev_test_train, format):#
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

#presents questions to the user so that they can enter concepts and their relationship (0 for inversely related and 1 for directly related)
def manually_annotate(dataset, name):
    path_to_file = os.path.join(OUTPUTPATH, f'conceptual_{name}.json')
    count = 0
    ten_entries = {}
    list_of_entries_and_concepts = []
    question_log = {}
    for i, entry in enumerate(dataset):
        if i > -1:
            similar_entry = similar_entry_exists(entry['question_mask'], entry['Option1'], entry['Option2'], question_log)
            if similar_entry:
                x, concepts, relation = similar_entry[0], similar_entry[1], similar_entry[2]
            else:
                concepts = enter_concepts(entry)
                relation = enter_relation(entry, concepts, question_log, i)
            ten_entries[i] = (entry, concepts, relation)
            count += 1
            if count % 10 == 0:
                print('########################################################################')
                for item in ten_entries:
                    print(item, '\n', ten_entries[item][0]['question'], '\n', ten_entries[item][0]['Option1'], ten_entries[item][0]['Option2'], ten_entries[item][0]['answer'], '\n', ten_entries[item][1], '\n', ten_entries[item][2])
                print('########################################################################')
                edit = 'start'
                while edit != 'ok':
                    while not edit.isnumeric() and edit != 'ok':
                        edit = input('Type the index of an entry to edit it. Type ok to confirm these entries')
                    if edit.isnumeric() and int(edit) in ten_entries:
                        cr = 'start'
                        while cr != '0' and cr != '1':
                            cr = input('edit concepts (0) or relations (1)?')
                        if cr == '0':
                            concepts = enter_concepts(ten_entries[int(edit)][0])
                            relations = enter_relation(ten_entries[int(edit)][0], concepts, question_log, int(edit))
                            ten_entries[int(edit)] = (ten_entries[int(edit)][0], concepts, relations)
                        if cr == '1':
                            relations = enter_relation(ten_entries[int(edit)][0], ten_entries[int(edit)][1], question_log, int(edit))
                            ten_entries[int(edit)] = (ten_entries[int(edit)][0], ten_entries[int(edit)][1], relations)
                        edit = 'start'
                        cr = 'start'
                        print('########################################################################')
                        for item in ten_entries:
                            print(item, '\n', ten_entries[item][0]['question'], '\n', ten_entries[item][0]['Option1'], ten_entries[item][0]['Option2'], ten_entries[item][0]['answer'], '\n', ten_entries[item][1], '\n', ten_entries[item][2])
                        print('########################################################################')
            #    elif edit == 'ok':
                indexical_order = sorted(ten_entries)
                with open(path_to_file, 'a') as f:
                    for index in indexical_order:
                        list_of_entries_and_concepts.append([index, ten_entries[index]])
                        print(f'{index}\t{ten_entries[index][0]["question"]} {ten_entries[index][0]["Option1"]} {ten_entries[index][0]["Option2"]}\t{ten_entries[index][1]}\t{ten_entries[index][2]}', file=f) 
                ten_entries = {}


def enter_concepts(entry):
    print(entry['question'])
    print(entry['Option1'], '###', entry['Option2'], entry['answer'])
    concepts = input('The (ideally two) concepts separated by a space:')
    concepts = concepts.split(' ')
    return concepts
        
def enter_relation(entry, concepts, question_log, i):
    relations = {}
    print(entry['question'])
    print(entry['Option1'], '###', entry['Option2'], entry['answer'])
    print(concepts)
    for i, c in enumerate(concepts[:-1]):
        for c2 in concepts[i+1:]:
            di = 'start'
            while di != '0' and di != '1':
                di = input('Are these directly (1) or inversely (0) related?')
            relations[c,c2] = di
    question_log[entry['question_mask'][:20], entry['Option1'], entry['Option2']] = i, concepts, relations
    return relations


#If a masked entry exists whose first 20 words match the current entry when it is masked, copy the first entry's conceptual information to the second.
def similar_entry_exists(question_masked, one, two, question_log):
    q = question_masked.split(' ')
    q.append(one)
    q.append(two)
    if (question_masked[:20], one, two) in question_log:
        return question_log[question_masked[:20], one, two]
    else:
        return False




#removes the most common nouns and prints the rest. Used to check which nouns are used in explanations. 
def check_data(dataset):
    uncommon_descriptors =[]
    count = 0
    for item in dataset:
        added = False
        idx, sentence_options, concepts, relation = item.split('\t')
        concepts = concepts.strip('][\'').split('\', \'')
        for concept in concepts:
            if concept not in ['time', 'distance', 'speed', 'weight', 'acceleration', 'heat', 'smoothness', 'brightness', 'loudness', 'mass', 'gravity', 'thickness', 'friction', 'temperature', 'ping'] and added == False:
                print(item)
                count += 1
                if concept not in uncommon_descriptors:
                    uncommon_descriptors.append(concept)
                added = True
    print(count)
    print(uncommon_descriptors)
                

def main():
    explanations = load_dataset('conceptual_data', CONCEPTUALFILES, 'txt')
    dev_explanations = explanations['dev']
    test_explanations = explanations['test']
    train_explanations = explanations['train']
    print('dev')
    check_data(dev_explanations)
    print('test')
    check_data(test_explanations)
    print('train')
    check_data(train_explanations)

main()
