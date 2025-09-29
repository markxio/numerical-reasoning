import json
import re
import os
from os import listdir
from os.path import isfile, join
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

from pydantic import BaseModel
from openai import OpenAI

from prompts import generate_qa_instruct
from split_sentences import pprint, get_files_in_folder

def main():
    # Load the configuration file
    with open("config.json") as f:
        config = json.load(f)["generate_question_answer"]

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

    sentences_folder = config["sentences_folder"]
    qa_folder = config["question_answer_folder"]
    check_already_processed = config["check_already_processed"]
    api_key_file = config["api_key_file"]

    with open(api_key_file) as f:
        config["API_KEY"] = json.load(f)["API_KEY"]
    
    #####################
    # split sentences 
    #####################

    print(f"Getting all files in folder {config['sentences_folder']} ...")
    sentences_files = get_files_in_folder(config['sentences_folder'])
    print(f"Number of files to process: {len(sentences_files)}")

    if check_already_processed:
        # SENTENCES_DOW_JONES/10-K/: 93410_10K_2023_0000093410-24-000013_item_9.txt
        # QA_DOW_JONES/:             93410_10K_2023_0000093410-24-000013_item_9_3.json # last _3 is the sentence id
        
        already_processed = get_files_in_folder(config['question_answer_folder'])
        df = pd.DataFrame({"filename": already_processed})
        df.to_csv("generate_task_1_already_prcoessed_sentences.csv", sep=";", index=False)

        sentences_files_not_yet_processed = []
        for sentence_txt_file in sentences_files:
            filename_no_extension = sentence_txt_file.replace(".txt", "")
            # check if substring exists in any of strings in list
            if any(filename_no_extension in mystring for mystring in already_processed):
                continue
            sentences_files_not_yet_processed.append(sentence_txt_file)
        sentences_files = sentences_files_not_yet_processed

    count_total=0
    os.system(f"mkdir -p {qa_folder}")
    for txt_file in tqdm(sentences_files):
        with open(f"{sentences_folder}/{txt_file}", "r") as f:
            count_file=0
            for idx, sentence in enumerate(f): # one sentence per line
                # move filters from filter_and_answer_tagging.py to here (at least for sentences)
                # for generated questions and answers, leave them in filter_and_answer_tagging.py

                # only rm whitespace and linebreaks really
                # as clean as it gets
                sentence_clean = sentence.replace("\n", "")
                if not starts_with_Aa(sentence_clean.replace(" ", ""), config):
                    # based on the regex's defined in config.json
                    #
                    # currently checks for an English sentence
                    # aka matches if the string starts with a capital letter,
                    # followed by a lowercase letter, and ending the string with
                    # with a fullstop
                    continue
                if not sentence_clean.replace(" ", "").endswith("."):
                    continue
                if len(sentence_clean.replace(" ", "")) < int(config['sentence_min_length']):
                    continue

                qa = send_request(system_content=generate_qa_instruct, user_content=sentence_clean, config=config, response_format=QuestionAnswer)
                # add sentence, txt_file (filename) to generated qa
                # txt_file (filename) includes company id and year
                # txt_file (filename) lets us trace back to original
                # edgar files

                # save qa+sentence to file
                target_file = f"{qa_folder}/{txt_file[:-4]}_{idx}.json"
                qa_json = qa.model_dump(mode="json")
                qa_json["sentence"] = sentence.replace("\n", "").strip()
                qa_json["src"] = f"{sentences_folder}/{txt_file}"
                #pprint(qa_json)
                #print(target_file)
                with open(target_file, "w") as f:
                    json.dump(qa_json, f)
                count_total = count_total + 1
                count_file = count_file + 1
        print(f"{txt_file} done -- {count_file} QA pairs generated")
        #exit("stopping after the first file for testing!")

    print(f"Finished, {count_total} QA-pairs generated!\n")

#def filter_sentence(sentence, config):
def starts_with_Aa(sentence, config):
    return bool(re.search(r"^[A-Z]{1}[a-z]{1}", sentence))

def get_unprocessed_files_only(left_dir: str, right_dir: str):
    # myfiles is a list of filenames (without path)
    # mydir is the directory holding files
    # we want to check if the filenames in myfiles is in mydir
    # only return a list of files, that are not in mydir yet
    left = get_files_in_folder(left_dir)
    right = get_files_in_folder(right_dir)
    # beware: set(left) - set(right) is not symmetric:
    # list(set([1,2]) - set([2,3])) == [1]
    return list(set(left) - set(right))    

# structured json output
# https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat
#
# Structured Outputs is available in our latest large language models, 
# starting with GPT-4o. 
# Older models like gpt-4-turbo and earlier may use JSON mode instead.
class QuestionAnswer(BaseModel):
    question: str
    answer: str

class Answer(BaseModel):
    answer: str

def send_request(system_content, user_content, config, response_format):

    # openai_api_key.json
    # gpt-4o

    # google_api_key.json
    # gemini-2.5-pro-preview-05-06

    if "google" in config["api_key_file"]:
        # use google gemini with openai python lib:
        # https://ai.google.dev/gemini-api/docs/openai
        client = OpenAI(
            api_key=config["API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        #completion = client.chat.completions.create( # google docs recommend this
        completion = client.beta.chat.completions.parse( # uses response_format
            model=config["model_name"],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=QuestionAnswer
        )
    elif "openai" in config["api_key_file"]:
        client = OpenAI(api_key=config["API_KEY"])
        #completion = client.chat.completions.create( 
        completion = client.beta.chat.completions.parse( # uses response_format
            model=config["model_name"],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=response_format #QuestionAnswer
        )
    else:
        raise ValueError("The api_key_file in config.json must be named google_api_key.json or openai_api_key.json to identify which client configuration to use (We can use the OpenAI python library to address the Google Gemini API.")

  
    """
    # example response for structured output with response_fromat
    {
      "steps": [
        {
          "explanation": "Start with the equation 8x + 7 = -23.",
          "output": "8x + 7 = -23"
        },
        {
          "explanation": "Subtract 7 from both sides to isolate the term with the variable.",
          "output": "8x = -23 - 7"
        },
        {
          "explanation": "Simplify the right side of the equation.",
          "output": "8x = -30"
        },
        {
          "explanation": "Divide both sides by 8 to solve for x.",
          "output": "x = -30 / 8"
        },
        {
          "explanation": "Simplify the fraction.",
          "output": "x = -15 / 4"
        }
      ],
      "final_answer": "x = -15 / 4"
    } 
    """
 
    #answer = completion.choices[0].message.content
    #print(answer)
    #return answer
    return completion.choices[0].message.parsed # returns json

if __name__=="__main__":
    main()
