#!/bin/python

import pandas as pd
import re
import ast
import argparse

from rouge import Rouge
rouge = Rouge()
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from nltk.translate import meteor

def calculate_rouge(candidate, reference):
    '''
    candidate, reference: generated and ground-truth sentences
    '''
    scores = rouge.get_scores([candidate], reference)
    return scores

def calculate_bleu(candidate, reference):
    '''
    candidate, reference: generated and ground-truth sentences
    '''
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    score = sentence_bleu(reference, candidate)
    return score

def calculate_meteor(candidate, reference):
  '''
  candidate, reference: tokenized list of words in the sentence
  '''
  reference = word_tokenize(reference)
  candidate = word_tokenize(candidate)
  meteor_score = round(meteor([candidate],reference), 4)
  return meteor_score

def metrics(candidate, reference):
    rouge_scores = calculate_rouge(candidate, reference)
    bleu_score = calculate_bleu(candidate, reference)
    meteor_score = calculate_meteor(candidate, reference)

    return rouge_scores, bleu_score, meteor_score

def eval(csv_test, csv_golden):
    golden_df = pd.read_csv(csv_golden, sep=";")
    test_df = pd.read_csv(csv_test, sep=";")
    
    # q1: financial metric
    # q2: years
    # q3: financial metric in year0
    # q4: financial metric change from year0 to year1 (increase, decrease, constant)
    # q5: by how much did financial metric change from year0 to year1 (numeric difference)

    evals_partial = []
    evals_partial.append([])
    evals_partial.append([])
    evals_partial.append([])
    evals_partial.append([])
    evals_partial.append([])

    evals_exact = []
    evals_exact.append([])
    evals_exact.append([])
    evals_exact.append([])
    evals_exact.append([])
    evals_exact.append([])
    for a1g, a2g, a3g, a4g, a5g, a1t, a2t, a3t, a4t, a5t in zip( \
                                    golden_df["task_1_answer"].tolist(), \
                                    golden_df["task_2_answer"].tolist(), \
                                    golden_df["task_3_answer"].tolist(), \
                                    golden_df["task_4_answer"].tolist(), \
                                    golden_df["task_5_answer"].tolist(), \
                                    test_df["task_1_answer"].tolist(), \
                                    test_df["task_2_answer"].tolist(), \
                                    test_df["task_3_answer"].tolist(), \
                                    test_df["task_4_answer"].tolist(), \
                                    test_df["task_5_answer"].tolist()):

        partial_financial_metric_match, \
            partial_years_match, \
            partial_value_year0_match, \
            partial_qual_change_match, \
            partial_quant_change_match = compare_partial(a1g, a2g, a3g, a4g, a5g, a1t, a2t, a3t, a4t, a5t)
        
        exact_financial_metric_match, \
            exact_years_match, \
            exact_value_year0_match, \
            exact_qual_change_match, \
            exact_quant_change_match = compare_exact(a1g, a2g, a3g, a4g, a5g, a1t, a2t, a3t, a4t, a5t)

        evals_partial[0].append(partial_financial_metric_match)
        evals_partial[1].append(partial_years_match)
        evals_partial[2].append(partial_value_year0_match)
        evals_partial[3].append(partial_qual_change_match)
        evals_partial[4].append(partial_quant_change_match)

        evals_exact[0].append(exact_financial_metric_match)
        evals_exact[1].append(exact_years_match)
        evals_exact[2].append(exact_value_year0_match)
        evals_exact[3].append(exact_qual_change_match)
        evals_exact[4].append(exact_quant_change_match)

        #q1scores = metrics(a1t, a1g)
        #q2scores = metrics(a2t, a2g)
        #q3scores = metrics(str(a3t), str(a3g))
        #q4scores = metrics(a4t, a4g)
        #q5scores = metrics(str(a5t), str(a5g))

    test_df_partial = test_df.copy()
    test_df_exact = test_df.copy()

    test_df_partial["task_1_eval"] = evals_partial[0]
    test_df_partial["task_2_eval"] = evals_partial[1]
    test_df_partial["task_3_eval"] = evals_partial[2]
    test_df_partial["task_4_eval"] = evals_partial[3]
    test_df_partial["task_5_eval"] = evals_partial[4]
    
    test_df_exact["task_1_eval"] = evals_exact[0]
    test_df_exact["task_2_eval"] = evals_exact[1]
    test_df_exact["task_3_eval"] = evals_exact[2]
    test_df_exact["task_4_eval"] = evals_exact[3]
    test_df_exact["task_5_eval"] = evals_exact[4]
    
    test_df_partial.to_csv(f"{csv_test[:-4]}_eval_partial.csv", sep=";", index=False)
    test_df_exact.to_csv(f"{csv_test[:-4]}_eval_exact.csv", sep=";", index=False)

    id_list = []
    n_list = []
    n_true_list = []
    ratio_true_list = []
    n_false_list = []
    ratio_false_list = []

    line_exact = f"{csv_test},"
    line_partial = f"{csv_test},"
    for i in range(1,6):
        n_true = len(list(test_df_partial[test_df_partial[f"task_{i}_eval"]==True].index))
        n_false = len(list(test_df_partial[test_df_partial[f"task_{i}_eval"]==False].index))
        n = len(list(test_df_partial.index)) 
        line_partial += "%.1f," % (100*n_true/n)

        n_true = len(list(test_df_exact[test_df_exact[f"task_{i}_eval"]==True].index))
        n_false = len(list(test_df_exact[test_df_exact[f"task_{i}_eval"]==False].index))
        n = len(list(test_df_exact.index)) 
        line_exact += "%.1f," % (100*n_true/n)

        id_list.append(f"task_{i}_eval")
        n_list.append(n)
        n_true_list.append(n_true)
        ratio_true_list.append(100 * n_true/n)
        n_false_list.append(n_false)
        ratio_false_list.append(100 * n_false/n)
    
    line_partial += "partial"
    line_exact += "exact"
    
    print(line_partial)
    print(line_exact)

    summary_df = pd.DataFrame({
        "id": id_list,
        "n": n_list,
        "n_true": n_true_list,
        "ratio_true": ratio_true_list,
        "n_false": n_false_list,
        "ratio_false": ratio_false_list
    })
    summary_df.to_csv(f"{csv_test[:-4]}_eval_summary.csv", sep=";", index=False, float_format='%.1f')

def compare_exact(a1g, a2g, a3g, a4g, a5g, a1t, a2t, a3t, a4t, a5t):
    ## Q1: FINANCIAL METRIC
    # compare two long string
    financial_metric_match = False
    if a1g == a1t:
        financial_metric_match = True

    ## Q2: YEARS
    # compare two lists
    years_match = False
    if a2g == a2t:
        years_match = True

    ## Q3: ENTITY EXTRACTION: VALUE IN YEAR0
    # $4.6 billion
    value_year0_match = False
    if a3g == a3t:
        value_year0_match = True

    ## Q4: QUALITATIVE: CHANGE FROM YEAR0 to YEAR1
    # increase, decrease, constant
    a4golden = a4g.split(",")[0]
    qual_change_match = False
    if a4golden == a4t:
        qual_change_match = True

    ## Q5: QUANTITATIVE: BY HOW MUCH DID IT CHANGE FROM YEAR0 to YEAR1
    # $4.6 billion
    a5golden = a5g.split(",")[0]
    #quant_change_match = compare_ints_from_str(str(a5t), str(a5g))
    quant_change_match = False
    if a5golden == a5t:
        quant_change_match = True

    return financial_metric_match, years_match, value_year0_match, qual_change_match, quant_change_match 

def compare_partial(a1g, a2g, a3g, a4g, a5g, a1t, a2t, a3t, a4t, a5t):
    ## Q1: FINANCIAL METRIC
    # compare two long strings
    # check if any subset is in the other string
    # all lowercase
    # only keep alpha characters (letters only, removes punctuation)
    alpha_test = re.sub(r'\W+', '', a1t.lower())
    alpha_golden = re.sub(r'\W+', '', a1g.lower())

    financial_metric_match = False
    if alpha_test in alpha_golden or alpha_golden in alpha_test:
        financial_metric_match = True

    ## Q2: YEARS
    # compare two lists
    years_golden = re.findall(r'\d{4}', a2g)
    years_test = re.findall(r'\d{4}', a2t)

    # typecast elements to int
    years_golden = sorted(list(map(int, years_golden)))
    years_test = sorted(list(map(int, years_test)))

    years_match = False
    if len(years_test) == len(years_golden):
        for yt, yg in zip(years_test, years_golden):
            if yt != yg:
                break
        years_match = True

    ## Q3: ENTITY EXTRACTION: VALUE IN YEAR0
    # $4.6 billion
    value_year0_match = compare_quantitative_val(a3t, a3g)

    ## get only digits from string and compare
    #value_year0_match = compare_ints_from_str(str(a3t), str(a3g))
    
    ## Q4: QUALITATIVE: CHANGE FROM YEAR0 to YEAR1
    # increase, decrease, constant
    a4golden = a4g.split(",")[0]
    #qual_change_match = compare_qual_answer(a4t, a4golden)
    qual_change_match = compare_qualitative(a4t, a4golden)

    ## Q5: QUANTITATIVE: BY HOW MUCH DID IT CHANGE FROM YEAR0 to YEAR1
    # $4.6 billion
    a5golden = a5g.split(",")[0]
    #quant_change_match = compare_ints_from_str(str(a5t), str(a5g))
    quant_change_match = compare_quantitative_val(a5t, a5golden)

    return financial_metric_match, years_match, value_year0_match, qual_change_match, quant_change_match

def get_val_from_dollars(s):
    no_whitespace = str(s).replace(" ", "")
    regex = r"\$(\d+\.{0,1}\d+)(million|billion|trillion|thousand)"
    matches = re.findall(regex, no_whitespace)
    if len(matches) == 0:
        return 0, 0
    return matches[0][0], matches[0][1] # access tuple in list

def compare_quantitative_val(test, golden):
    dollars_val_golden, dollars_slug_golden = get_val_from_dollars(golden)
    dollars_val_test, dollars_slug_test = get_val_from_dollars(test)
    if dollars_val_golden == dollars_val_test:
        return True
    else:
        # check if they are off by a factor of 10
        if dollars_slug_golden == "billion" and dollars_slug_test == "million":
            if float(dollars_val_golden) * 100 == float(dollars_val_test):
                return True
    return False

def compare_ints_from_str(test, golden):
    test_tmp = re.findall(r"\d", test)
    #test_tmp = "".join(test_tmp)
    golden_tmp = re.findall(r"\d", golden)
    golden_tmp = "".join(golden_tmp)

    #match = True if test_tmp == golden_tmp else False
    #return match   

    for d in test_tmp:
        if d not in golden_tmp:
            return False
    return True

def compare_qualitative(test, golden):
    regex = r"(increase|decrease|constant)"
    test_tmp = re.findall(regex, test.lower())
    golden_tmp = re.findall(regex, golden.lower())
    if len(test_tmp) == 0 or len(golden_tmp) == 0:
        return False
    elif test_tmp[0] == golden_tmp[0]:
        return True
    elif test_tmp[0][0] in golden_tmp[0][0] or golden_tmp[0] in test_tmp[0]:
        return True
    else:
        return False

def compare_qual_answer(test, golden):
    test_tmp = test.replace(" ", "").lower()
    test_tmp = test_tmp.split(",")[0].strip()
    golden_tmp = golden.replace(" ", "").lower()
    golden_tmp = golden_tmp.split(",")[0].strip()
    
    #if len(test_tmp) != len(golden_tmp):
    #    return False

    #for t, g in zip(test_tmp, golden_tmp):
    #    if t != g:
    #        return False

    #return True

    if test_tmp == golden_tmp:
        return True
    return False

def liststr_to_argstr(list_str):
    try:
        list_out = ast.literal_eval(list_str)
    except:
        # not interpretable by ast as a list
        if "," in list_str:
            list_out = list_str.split(",")
        elif "and" in list_str:
            list_out = list_str.split("and")
        else:
            list_out = []
    return list_out 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_golden", type=str, help="CSV file with golden answers")
    parser.add_argument("csv_test", type=str, help="CSV file with model answers")

    args = parser.parse_args()
    csv_test = args.csv_test
    csv_golden = args.csv_golden

    eval(csv_test, csv_golden)
