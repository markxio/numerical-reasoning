import json
import os
import re
from os import listdir
from os.path import isfile, join
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '../edgar-crawler')
from item_lists import item_list_10k

from nltk import tokenize
download_dir="."
os.system("mkdir -p NLTK_DATA")
os.environ["NLTK_DATA"] = f"{download_dir}/NLTK_DATA"

# download on login node once, to the dir specified above
# on compute node without internet connection, just specify the env var
# and disable the download (we use the cached dir then)
#import nltk
#nltk.download('punkt', download_dir=download_dir)

import spacy

"""
    json example
    ----------------

    "cik": "789019",
    "company": "MICROSOFT CORP",
    "filing_type": "10-K",
    "filing_date": "2024-07-30",
    "period_of_report": "2024-06-30",
    "sic": "7372",
    "state_of_inc": "WA",
    "state_location": "WA",
    "fiscal_year_end": "0630",
    "filing_html_index": "https://www.sec.gov/Archives/edgar/data/789019/0000950170-24-087843-index.html",
    "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
    "complete_text_filing_link": "https://www.sec.gov/Archives/edgar/data/789019/0000950170-24-087843.txt",
    "filename": "789019_10K_2024_0000950170-24-087843.htm",
    "item_1": "...",
    ...
    "item_16": "..."

"""

def main():
    # Load the configuration file
    with open("config.json") as f:
        config = json.load(f)["split_sentences"]

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

    raw_filings_folder = config["raw_filings_folder"]
    enable_threading = config["enable_threading"]
    cores = int(config["parallel_processes"])
    sentences_folder = config["sentences_folder"]
    sentence_min_lenght = config["sentence_min_length"]
    labels_to_include = config["spacy_labels_to_include"]

    #####################
    # split sentences 
    #####################

    print(f"Getting all files in folder {config['raw_filings_folder']} ...")
    raw_files = get_files_in_folder(config['raw_filings_folder'])
    print(f"Number of files to process: {len(raw_files)}")

    if enable_threading:
        # split list of raw_files into n=cores chunks
        chunks = np.array_split(raw_files, cores)
        Parallel(n_jobs=cores)(delayed(process_files)(chunks[i], config) for i in range(0, cores))
    else:
        process_files(raw_files, config)

    print(f"Finished processing!\n")

def process_files(raw_files, config):
    os.system(f"mkdir -p {config['sentences_folder']}") 
    for raw_file in tqdm(raw_files):
        # load_json
        json_file = f"{config['raw_filings_folder']}/{raw_file}" 
        with open(json_file) as f:
            obj = json.load(f)
            
        # iterate through json obj
        # check if json obj key in item list: from 1 to 16 (note 2a, 2b etc)
        for key, value in obj.items():
            item = key.replace("item_", "")
            if item.upper() in item_list_10k:
                        
                # get text for each item
                paragraph = obj[f"item_{item}"]
                sentences = split_into_sentences(paragraph)

                # save list of sentences to file {original_name}_item_{i}.txt
                target_file = f"{raw_file.split('.')[0]}_item_{item.lower()}.txt" # rm file extension and save as .txt file
                
                numeric_pattern=r"\d"
                with open(f"{config['sentences_folder']}/{target_file}", 'w') as f:
                    for sentence in sentences:
                        
                        # filter for numeric values e.g., %, \d, what else?
                        if len(sentence.replace(" ", "")) < int(config['sentence_min_length']):
                            continue
                        if not has_numbers(remove_dates(sentence)):
                            continue
                        if spacy_match(sentence, config['spacy_labels_to_include']):
                            # write sentence to file (one line per sentence)
                            f.write(f"{sentence}\n")


def spacy_match(sentence, labels_to_include):
    # returns true if at least a single label of labels_to_include
    # was found in the sentence

    # install spacy:
    # pip install spacy (complains about numpy at wrong version)
    # pip install pipenv
    # pipenv install spacy
    # pipenv sync --system --dev (now spacy should be installed and available)

    # dont forget to download the model first:
    # python -m spacy download en_core_web_trf (roberta-base transformer) 
    # python -m spacy download en_core_web_sm (English pipeline optimized for CPU)

    # to download to specific path:
    # python -m spacy download en_core_web_sm --data-path $download_dir

    # English transformer pipeline (Transformer(name=‘roberta-base’, piece_encoder=‘byte-bpe’, stride=104, type=‘roberta’, width=768, window=144, vocab_size=50265))
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc.ents:
        if token.label_ in labels_to_include:
            return True
    return False

def remove_dates(mystr):
    # only substitutes dates of pattern "November 24, 2024"
    pattern=r"[a-zA-Z]+\s+\d{1,2},\s+\d{4}"
    replaced = re.sub(pattern, "", mystr)
    return replaced

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def split_into_sentences(paragraph):
    return tokenize.sent_tokenize(paragraph)

def get_files_in_folder(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def pprint(mydict: dict):
    print(json.dumps(mydict, sort_keys=True, indent=4))

if __name__=="__main__":
    main()
