import os
import re
import json
from joblib import Parallel, delayed
import spacy
from tqdm import tqdm
import numpy as np
import pandas as pd

from split_sentences import pprint, get_files_in_folder

def main():
    # Load the configuration file
    with open("config.json") as f:
        config = json.load(f)["filter"]

    print("-------------------------------")
    print("---------- config -------------")
    print("-------------------------------")
    pprint(config)
    print("-------------------------------")
    print("-------------------------------")
    print("-------------------------------")

    #####################
    # set config args
    #####################

    qa_folder = config["question_answer_folder"]
    filter_folder = config["filter_folder"]
    sentence_filters = config["sentence_filters"]
    answer_max_length = config["answer_max_length"]
    counter_map_file = config["counter_map_file"]
    label_map_file = config["label_map_file"]
    label_map_counts_only_file = config["label_map_counts_only_file"]
    enable_threading = config["enable_threading"]
    cores = int(config["parallel_processes"])

    #####################
    # split sentences 
    #####################

    print(f"Getting all files in folder {config['question_answer_folder']} ...")
    qa_json_files = get_files_in_folder(config['question_answer_folder'])
    print(f"Number of json files (generated QA-pairs): {len(qa_json_files)}")

    os.system(f"rm -rf {filter_folder}")
    os.system(f"mkdir -p {filter_folder}")
  
    counter_map = {}
    label_map = {} 
    filenames = []
    tickers = []
    years = []
    accepted = 0
    rows = []

    if enable_threading:
        chunks = np.array_split(qa_json_files, cores)
        result_chunks = Parallel(n_jobs=cores)(delayed(filter_triplets)(chunks[i], config) for i in range(0, cores))
        
        # merge partial maps from Parallel's result_chunks
        for result in result_chunks:
            for ticker, mydict in result[0].items(): # counter_map
                if ticker in counter_map.keys():
                    for year, count in mydict.items():
                        if year in counter_map[ticker].keys():
                            current_count = counter_map[ticker][year]
                            counter_map[ticker][year] = current_count + count
                        else:
                            counter_map[ticker][year] = count
                else:
                    counter_map[ticker] = mydict
            for label, mydict in result[1].items(): # label_map
                if label in label_map.keys():
                    current_count = label_map[label]["count"]
                    label_map[label]["count"] = current_count + mydict["count"]
                    current_list = label_map[label]["answers"]
                    current_list.extend(mydict["answers"])
                    label_map[label]["answers"] = current_list
                else:
                    label_map[label] = {}
                    label_map[label]["count"] = mydict["count"]
                    label_map[label]["answers"] = mydict["answers"]
            # .append() returns None
            # .extend() returns the new list
            filenames.extend(result[2])
            tickers.extend(result[3])
            years.extend(result[4]) 
            accepted = accepted + result[5]
            rows.extend(result[6])            
    else:
        counter_map, label_map, filenames, tickers, years, accepted, rows = filter_triplets(qa_json_files, config)

    label_map_counts_only = {}
    for label, mydict in label_map.items():
        label_map_counts_only[label] = mydict["count"]

    # write to file
    with open(f"{config['label_map_file']}", "w") as f:
        json.dump(label_map, f)

    with open(f"{config['label_map_counts_only_file']}", "w") as f:
        json.dump(label_map_counts_only, f)

    with open(f"{config['counter_map_file']}", "w") as f:
        json.dump(counter_map, f)

    print(f"QA-pairs filtered, to be found in folder {config['filter_folder']}.")
    print(f"N={accepted} accepted triplets (sentence, question, answer)")
    unique_src_files = list(set(filenames))
    print(f"Number of 10-K filings: {len(unique_src_files)}")
    
    unique_tickers = list(set(tickers))
    print(f"Number of tickers (unique ticker per company): {len(unique_tickers)}")

    unique_years = list(set(years))
    sorted_years = sorted(unique_years)
    if len(sorted_years) > 0:
        print(f"Number of years: {len(unique_years)} ({sorted_years[0]} - {sorted_years[len(sorted_years)-1]})")

    #data = {
    #    "filepath": "",
    #    "year": "",
    #    "ticker": "",
    #    "10k_item": "",
    #    "10k_sentence_id": "",
    #    "passes_quality_filter": "",
    #    "answer_labels": "",
    #    "sentence": "",
    #    "question": "",
    #    "answer": ""
    #}

    #row = [
    #    f"{filepath}",
    #    f"{year}",
    #    f"{ticker}",
    #    f"{10k_item}",
    #    f"{10k_sentence_id}",
    #    f"{passes_quality_filter}",
    #    f"{answer_labels}",
    #    f"{sentence}",
    #    f"{question}",
    #    f"{answerf}" 
    #]

    #rows=[]
    #rows.append(row)
    cols = [
            "json_file_path", # the qa-sentence triplet json file
            "year", # year of the filing
            "ticker_id", # unique company id used on edgar
            "item", # section of the 10-k filing document
            "sentence_id", # unique per 10-k filing document
            "passes_quality_filter", # passed our quality filters above or not
            "answer_labels", # spacy tagged labels of the answer only
            "sentence", # sentence text (from the original filing section, separated with nltk.tokenize)
            "question", # the gpt-4o generated question (see prompts.py)
            "answer" # the gpt-4o generated answer to above's question (see prompts.py)
    ]
    df = pd.DataFrame(rows, columns=cols)
    # replaced semicolons with colons
    # so save to use semicolon as separator here
    df.to_csv("FILTER.csv", sep=";", index=False)

def filter_triplets(qa_json_files, config):
    # example
    # SENTENCES_DOW_JONES/10-K/93410_10K_2017_0000093410-18-000010_item_14_839.json  
    filenames = []
    tickers = []
    years = []
    #counter_map = {
    #    f"{ticker}": {
    #        f"{year}": f"{count_per_ticker_per_year",
    #    },
    #}
    counter_map = {}
    #label_map = {
    #    f"{label}": f"{count}"
    #}
    label_map = {}
    exactly_one_label = 0
    exactly_one_label_files = []
    rows = []

    count_accepted=0
    for json_file in tqdm(qa_json_files):
        json_file_path = f"{config['question_answer_folder']}/{json_file}"
        with open(json_file_path) as f:
            qa = json.load(f)

        # path + filename
        # get everything after last slash (aka filename)
        src_file = qa["src"].split("/")[-1]
        
        item_pattern = r"_item_(\d+[a-z]{0,})_(\d+)\.json" 
        match_obj = re.search(item_pattern, json_file)
        item = ""
        sentence_id = -1
        if match_obj:
            item = match_obj.group(1)
            sentence_id = match_obj.group(2)

        # sentence source file is a txt file
        # with one sentence per line (ending lines with \n)
        src_file_item_pattern = r"_item_(\d+[a-z]{0,})\.txt" 
        src_file = re.sub(item_pattern, "", src_file)

        sentence = qa["sentence"].strip().replace(";", ":")
        question = qa["question"].strip().replace(";", ":") 
        answer = qa["answer"].strip().replace(";", ":") 

        if "cannot" in answer:
            # from prompt instructions:
            #
            # If you are not certain, then output 
            # the following phrase as answer: 
            # 'Answer cannot be determined.'
            continue
        if len(answer.replace(" ", "")) > int(config['answer_max_length']):
            continue
      
        answer_labels = spacy_label(answer)
        for label in answer_labels:
            if label in label_map.keys():
                # increase existing count
                # add filename to existing list
                label_map[label]["count"] = label_map[label]["count"] + 1

                # careful! Dont:
                # label_map[label]["answers"] = label_map[label]["answers"].append(json_file_path)
                current_list = label_map[label]["answers"]
                current_list.append(json_file_path)
                label_map[label]["answers"] = current_list
            else:
                # create new entry in map
                label_map[label] = {}
                label_map[label]["count"] = 1
                label_map[label]["answers"] = [ json_file_path ]
            
        if len(answer_labels) == 1:
            exactly_one_label = exactly_one_label + 1
            exactly_one_label_files.append(json_file_path) 
            
            if "MONEY" in answer_labels:
                # create new csv file only includes
                # samples with sentence mentioning MONEY twice
                # aka find $ symbol exactly twice
                # 
                # could do the same for PERCENT...
                pass

        count_accepted = count_accepted + 1

        with open(f"{config['filter_folder']}/{json_file}", "w") as f:
            json.dump(qa, f)

        filenames.append(src_file)

        splits = src_file.split("_")
        ticker = splits[0]
        year = splits[2]

        tickers.append(ticker)
        years.append(year)

        # add count to map
        # counts number of QA-pairs per ticker and per year
        if ticker in counter_map.keys():
            if year in counter_map[ticker].keys():
                current_count = counter_map[ticker][year]
                counter_map[ticker][year] = current_count + 1
            else:
                counter_map[ticker][year] = 1
        else:
            counter_map[ticker] = {}
            counter_map[ticker][year] = 1

        passes_quality_filter=True
        row = [
            f"{json_file_path}", # the qa-sentence triplet json file
            f"{year}", # year of the filing
            f"{ticker}", # unique company id used on edgar
            f"{item}", # section of the 10-k filing document
            f"{sentence_id}", # unique per 10-k filing document
            f"{passes_quality_filter}", # passed our quality filters above or not
            f"{answer_labels}", # spacy tagged labels of the answer only
            f"{sentence}", # sentence text (from the original filing section, separated with nltk.tokenize)
            f"{question}", # the gpt-4o generated question (see prompts.py)
            f"{answer}" # the gpt-4o generated answer to above's question (see prompts.py)
        ]
        rows.append(row)        

    label_map["EXACTLY_ONE_LABEL"] = {}
    label_map["EXACTLY_ONE_LABEL"]["count"] = exactly_one_label
    label_map["EXACTLY_ONE_LABEL"]["answers"] = exactly_one_label_files
    
    return counter_map, label_map, filenames, tickers, years, count_accepted, rows 

def spacy_label(answer):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(answer)
    labels = []
    for token in doc.ents:
        labels.append(token.label_)
    return labels

if __name__=="__main__":
    main()
