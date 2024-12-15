import os
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import sklearn
import numpy as np
PATH = 'results'


#collects information about predictions, and groups together the predictions of similar sentences.
def collect(names):
    predictions_main_dict = {}
    frequency_dict = {}
    for name in names:
        print(name)
        predictions_sublist =[]
        masked_questions = []
        
        with open(os.path.join(PATH, name)) as f:
            for line in f:
                masked_q = []
                line = line.strip('\'\"()\n')
                line = line.split(',')
                label = line[-1]
                label = int(label)
                prediction = line[-2]
                prediction = int(prediction)
                predictions_sublist.append(prediction)
                question_and_choices = line[:-2]
                question_and_choices = ''.join(question_and_choices)
                question_and_choices = word_tokenize(question_and_choices)
                for word in question_and_choices:
                    try:
                        word = int(word)
                        masked_q.append('[MASK]')
                    except:
                        masked_q.append(word)
                masked_q = ' '.join(masked_q)
                masked_questions.append([masked_q, prediction, label])
                if masked_q[:40] in frequency_dict:
                    if prediction == label:
                        frequency_dict[masked_q[:40]][0] +=1
                    else:
                        frequency_dict[masked_q[:40]][1] +=1
                else:
                    if prediction == label:
                        frequency_dict[masked_q[:40]] = [1, 0]
                    else:
                        frequency_dict[masked_q[:40]] = [0, 1]
        pred_array = np.asarray(predictions_sublist)
        predictions_main_dict[name] = pred_array
    with open('freq_math.txt', 'w') as f2:
        for item in frequency_dict:
            print(item, frequency_dict[item], file=f2)  
        
    return frequency_dict, masked_questions, predictions_main_dict

#calculates Cohen's Kappa between all model combinations
def c_kappa(math_dict, base_dict):
    base_dict.update(math_dict)
    all_dict = base_dict
    end = len(all_dict.keys())-1
    all_kappas = {}
    lst = list(all_dict.keys())
    for i, k in enumerate(lst):
        if i != end:
            for k2 in lst[i+1:]:
                if k == 'base_c_no_dev_explanations_result_data.txt' or k == 'math_c_no_dev_explanations_result_data.txt' or k2 == 'base_c_no_dev_explanations_result_data.txt' or k2 == 'math_c_no_dev_explanations_result_data.txt': 
                    all_kappas[(k,k2)] = sklearn.metrics.cohen_kappa_score(all_dict[k], all_dict[k2])
    
    with open('kappas.txt', 'w') as f:
        for k3 in all_kappas:
            print(k3, all_kappas[k3])
                             

def main():
    base = ['base_none_no_dev_explanations_result_data.txt', 'base_n_no_dev_explanations_result_data.txt', 'base_m_no_dev_explanations_result_data.txt', 'base_c_no_dev_explanations_result_data.txt', 'base_nm_no_dev_explanations_result_data.txt', 'base_nmc_no_dev_explanations_result_data.txt']
    math = ['math_none_no_dev_explanations_result_data.txt', 'math_n_no_dev_explanations_result_data.txt', 'math_m_no_dev_explanations_result_data.txt', 'math_c_no_dev_explanations_result_data.txt', 'math_nm_no_dev_explanations_result_data.txt', 'math_nmc_no_dev_explanations_result_data.txt']
    freq_dict_base, masked_base, predictions_dict_base = collect(base)
    freq_dict_math, math_base, predictions_dict_math = collect(math)
    c_kappa(predictions_dict_math, predictions_dict_base)


main()