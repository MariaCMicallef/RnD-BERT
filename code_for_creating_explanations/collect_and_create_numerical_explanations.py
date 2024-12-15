#There were 2 versions of this file
#In the new version, I have made a file numerical_leftovers.txt with entries where I couldn't extract the numbers automatically. add_numerical_manual_explanations loads them into the py file, and numbers always uses the numbers in that file for entries whose index are listed in it. Entries with 1 number or less are skipped. The py file now outputs the datasets with numerical explanations added. Numerical explanations have the format "3 is smaller than 25, bigger than 0.18 and the same number as 3, 25 is bigger than 0.18 and bigger than 3 and 0.18 is smaller than 3." or "12 is bigger than 10.".
#In the old version, numbers() went through the data and build numerical explanations in the format 'num1 is smaller than num2.'. If it didn't find at least 2 numbers in an entry, it printed that entry in a file called fmissing, and calls check_possible_answer_numbers, which if it found additional numbers in the options,  printed them in foptions. This information was used to create numerical_leftovers.txt
import json
import os
import nltk
#nltk.download('punkt')
from nltk import word_tokenize

INPUTPATH = 'clean_data'
OUTPUTPATH = 'numerical_data'
#stuff outputted into OUTPUTPATH: datasets with numerical explanations added, txt files of the questions with no explanations, txt files of the questions to which explanations have been added but which have new numbers in the possible answers (these files should be empty except for header titles)

def load_dataset(path):
    datasets_dict = {}
    names = ['QQA_dev_clean.json', 'QQA_test_clean.json', 'QQA_train_clean.json']
    labels = ['dev', 'test', 'train']
    for i, name in enumerate(names):
        file_path = os.path.join(path, name)
        f = open(file_path)
        datasets_dict[labels[i]] = json.load(f)
    return datasets_dict


def check_possible_answer_numbers(numbers_in_question, option_1_list, option_2_list):
    #receives the numbers found in a question (only questions where 2 or more numbers have been found should be sent here). Checks if there are any numbers in the possible options that are not in the question. If there are, returns the list of numbers. Otherwise, returns false.
    options_list = []
    new_numbers = []
    for lst in [option_1_list, option_2_list]:
        for word in lst:
            try:
                float_word = float(word)
                options_list.append(word)
            except:
                pass
    if not options_list:
        return False
    else:
        for num in options_list:
            if num not in numbers_in_question:
                new_numbers.append(num)
        if new_numbers:
            return new_numbers
        else:
            return False

def numbers(dataset, name, manual_additions):
    #Goes through each dataset and gets all the numbers. If there are 2, formulates an explanation, adds it to the json dataset, and dumps it in a new file called numerical_train.json or equivalent. If not, adds a placeholder instead of an explanation (f'MISSING_NUMERICAL_{i}') and keeps note of the question and the numbers gotten from it, and prints them in a file called mistakes_numerical
    number_list = [] #all the numbers of all the entries, with empty lists if not 2 numbers
    mistake_list = [] #where not 2 numbers in an entry, put the question and numbers here.
    no_explanation = 0
    count = 0
    correct_but_new_numbers_in_options = [['index', 'question', 'option1', 'option2', 'explanation', 'numbers in question', 'numbers in options']]
    for i, entry in enumerate(dataset):
        entry_list = [] #the two (or more or less) numbers in the question
        question_string = entry['question']
        option_1_string = entry['Option1']
        option_2_string = entry['Option2']
        question_list = word_tokenize(question_string) #question_string.split(' ')
        option_1_list = word_tokenize(option_1_string)
        option_2_list = word_tokenize(option_2_string)
        for word in question_list:
            try:
                float_word = float(word)
                entry_list.append(word)
            except:
                pass
        if str(i) in manual_additions:
            entry_list = manual_additions[str(i)]
        if entry_list != None:
            #Creates explanations comparing 2 numbers
            if len(entry_list) == 2:
                number_list.append(entry_list)
                num_1 = entry_list[0]
                num_2 = entry_list[1]
                #deals with one number which was raising an error
                if num_2 == '1/10':
                    num_2 = 0.1
                if float(num_1) > float(num_2):
                    new_data = f'{entry_list[0]} is bigger than {entry_list[1]}.'
                elif float(num_1) < float(num_2):
                    new_data = f'{entry_list[0]} is smaller than {entry_list[1]}.'
                else:
                    new_data = f'{entry_list[0]} is the same number as {entry_list[1]}.'
                entry['numerical_explanation'] = new_data #numerical_comparison 
                count +=1
                new_nums = check_possible_answer_numbers(entry_list, option_1_list, option_2_list)
                if new_nums:
                    correct_but_new_numbers_in_options.append([i, question_string, option_1_string, option_2_string, new_data, entry_list, new_nums])
            #creates explanations comparing 3 or more numbers.
            elif len(entry_list) > 2:
                entry_number_list = []
                for i, num in enumerate(entry_list[:-1]):
                    one_comparison = []
                    start = f'{num} is '
                    for num2 in entry_list[i+1:]:
                        if float(num) > float(num2):
                            one_comparison.append(f'bigger than {num2}')
                        elif float(num) < float(num2):
                            one_comparison.append(f'smaller than {num2}')
                        else:
                            one_comparison.append(f'the same number as {num2}')
                    if len(one_comparison) > 1:
                        number_phrase = ', '.join(one_comparison[:-1])
                        number_phrase = f'{start}{number_phrase} and {one_comparison[-1]}'
                    else:
                        number_phrase = f'{start}{one_comparison[-1]}'
                    entry_number_list.append(number_phrase)
                    #checks for numbers in the given possible answers that are not in the question. If yes, prints them in a file. (There are none.) This part doesn't need to be run now, since we've ascertained there are no such questions.
                    new_nums = check_possible_answer_numbers(entry_list, option_1_list, option_2_list)
                    if new_nums:
                        correct_but_new_numbers_in_options.append([i, question_string, option_1_string, option_2_string, new_data, entry_list, new_nums])
                #this part joins the different phrases comparing two numbers together into one sentence.
                number_sent = ', '.join(entry_number_list[:-1])
                number_sent += f' and {entry_number_list[-1]}.'
                entry['numerical_explanation'] = number_sent
                count += 1
            else:#If only one or no numbers are found
                print(i, question_string, entry_list, entry['Option1'], entry['Option2'])
                raise Exception
                number_list.append([])
                mistake_list.append([i, question_string, entry_list, entry['Option1'], entry['Option2']])
                #entry['numerical_explanation'] = f'MISSING_NUMERICAL_{i}' #This part is commented out. It was used to keep track of the places in the json file with a missing explanation. However I will simply not be including an explanation where I cannot make one in the format of the rest of the explanations.
        else:
            no_explanation += 1
    #creates paths so that the files are made in their own directory, making things more organized
    numerical_path = os.path.join(OUTPUTPATH, f'finalnumerical_{name}.json')#os.path.join(OUTPUTPATH, name)
    missing_numerical_path = os.path.join(OUTPUTPATH, f'missing_numerical{name}.txt')
    new_numbers_in_options_path = os.path.join(OUTPUTPATH, f'new_numbers_in_options{name}.txt')
    #Creates json files with numerical explanations
    with open(numerical_path, 'w') as f2:
        json.dump(dataset, f2)
    #Creates txt files with the questions with no explanation. Should now not include the ones where I manually identified the numbers, only the ones with really no explanation.
    with open(missing_numerical_path, 'w') as fmissing:
        for item in mistake_list:
            print(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\t{item[4]}', file=fmissing)
    #prints the question where an explanation has been made but there are additional numbers in the possible answers. This is always True because of the heading titles that are included in the list, but it prints an empty file.
    if correct_but_new_numbers_in_options:
        with open(new_numbers_in_options_path, 'w') as foptions:
            for item2 in correct_but_new_numbers_in_options:
                print(f'{item2[0]}\t{item2[1]}\t{item2[2]}\t{item2[3]}\t{item2[4]}\t{item2[5]}\t{item2[6]}', file=foptions)
    else:
        print('no new numbers')
                                
    return number_list, mistake_list, dataset, count, no_explanation

def add_numerical_manual_explanations()
    manual_additions_path = os.path.join(INPUTPATH, 'numerical_leftovers.txt')
    with open(manual_additions_path) as manual_number_file:
        train_test_or_dev = None
        train = {}
        dev = {}
        test = {}
        for line in manual_number_file:
            line = line.rstrip().split('\t')
            if len(line) == 1 and line[0] in ['train', 'dev', 'test']:
                if line[0] == 'train':
                    train_test_or_dev = 'train'
                elif line[0] == 'test':
                    train_test_or_dev = 'test'
                elif line[0] == 'dev':
                    train_test_or_dev = 'dev'
            else:
                index = line[0]
                if train_test_or_dev == 'train':
                    if len(line) > 1:
                        train[index] = line[1:]
                    else:
                        train[index] = None
                elif train_test_or_dev == 'test':
                    if len(line) > 1:
                        test[index] = line[1:]
                    else:
                        test[index] = None
                if train_test_or_dev == 'dev':
                    if len(line) > 1:
                        dev[index] = line[1:]
                    else:
                        dev[index] = None
        numerical_manual_additions = [dev, test, train]
        return numerical_manual_additions
        
    
def main():
    datasets = load_dataset(INPUTPATH)
    numerical_manual_additions = add_numerical_manual_explanations()#loads numbers for entries that are difficult to obtain automatically from the manually created file
    train_numbers, train_numbers_mistakes, train, train_correct, train_incorrect = numbers(datasets['train'], 'train', numerical_manual_additions[2])
    print('train correct numbers', train_correct, 'mistaken numbers', train_incorrect, 'total numbers', len(datasets['train']), 'unaccounted for', len(datasets['train'])- train_correct - train_incorrect)
    dev_numbers, dev_numbers_mistakes, dev, dev_correct, dev_incorrect = numbers(datasets['dev'], 'dev', numerical_manual_additions[0])
    print('dev correct numbers', dev_correct, 'mistaken numbers', dev_incorrect, 'total numbers', len(datasets['dev']), 'unaccounted for', len(datasets['dev'])- dev_correct -dev_incorrect)
    test_numbers, test_numbers_mistakes, test, test_correct, test_incorrect = numbers(datasets['test'], 'test', numerical_manual_additions[1])
    print('test correct numbers', test_correct, 'mistaken numbers', test_incorrect, 'total numbers', len(datasets['test']), 'unaccounted for', len(datasets['test'])- test_correct -test_incorrect)



main()

