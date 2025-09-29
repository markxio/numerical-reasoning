#!/bin/python

import json
import re
import os
from os import listdir
from os.path import isfile, join
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import ast

from pydantic import BaseModel
from openai import OpenAI

from prompts import generate_qa_instruct
from split_sentences import pprint, get_files_in_folder

def get_context_specific_questions(current_financial_metric, current_years):
    q2 = f"Did {current_financial_metric} from {current_years[0]} to {current_years[1]} increase, decrase or stay constant?"
    q3 = f"By how much did {current_financial_metric} change from {current_years[0]} to {current_years[1]}?"
    return q2, q3

def postprocess():
    dataset = pd.read_csv("dataset_context_specific_questions.csv", sep=";")
    # cols
    # json_file_path;year;ticker_id;item;sentence_id;answer_labels;sentence;task_1_question;task_1_answer;task_2_answer_qualitative;task_2_answer_quantitative;years;financial_metric;task_2_question;task_3_question
    dataset = dataset.drop(columns=['task_2_question', 'task_3_question'])
    q2_list = []
    q3_list = []
    for financial_metric, years in zip(tqdm(dataset["financial_metric"]), dataset["years"]):
        years_list = ast.literal_eval(years)
        q2, q3 = get_context_specific_questions(financial_metric, years_list)
        q2_list.append(q2)
        q3_list.append(q3)
    dataset["task_2_question"] = q2_list
    dataset["task_3_question"] = q3_list
    dataset.to_csv("dataset_context_specific_questions_fixed.csv", sep=";", index=False)
    
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

    api_key_file = config["api_key_file"]

    with open(api_key_file) as f:
        config["API_KEY"] = json.load(f)["API_KEY"]

    ######################
    ### start logic
    ######################

    # get question 1 from dataset
    dataset = pd.read_csv("dataset.csv", sep=";")
    # cols:
    # json_file_path;year;ticker_id;item;sentence_id;answer_labels;sentence;task_1_question;task_1_answer;task_2_answer_qualitative;task_2_answer_quantitative
    years = []
    financial_metrics = []
    context_specific_q2 = []
    context_specific_q3 = []
    for sentence in tqdm(dataset["sentence"]):
        current_years = find_years_in_sentence(sentence)
        years.append(current_years)

        if years == None:
            years.append(0)
            financial_metrics.append(0)
            context_specific_q2.append(0)
            context_specific_q3.append(0)
            continue
            
        # what is the financial metric in the sentence?
        instruct = "The following sentence stems from a SEC 10-K annual filing of a company. "
        instruct += "What is the financial metric mentioned in the sentence? Name the full name of the metric as provided in the sentence."
        a = send_request(system_content=instruct, user_content=sentence, config=config, response_format=Answer)
        current_financial_metric = a.answer
        financial_metrics.append(current_financial_metric)

        q2, q3 = get_context_specific_questions(current_financial_metric, current_years)

        context_specific_q2.append(q2)
        context_specific_q3.append(q3)

    dataset["years"] = years
    dataset["financial_metric"] = financial_metrics
    dataset["task_2_question"] = q2
    dataset["task_3_question"] = q3

    dataset.to_csv("dataset_context_specific_questions.csv", sep=";", index=False)
    # get financial metric from question 1
    # get years in sentence
    # construct context-specific questions Q2 and Q3 based on metric from Q1 + years from sentence

def find_years_in_sentence(sentence):
    occurrences = re.findall(r"\d{4}", sentence)
    years = []
    for potential_year in occurrences:
        if int(potential_year) >= 1993 and int(potential_year) <= 2024:
            years.append(int(potential_year))
        
    if 1 == sorted(years)[1] - sorted(years)[0]:
        return sorted(years)

    return None

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

    return completion.choices[0].message.parsed # returns json

if __name__=="__main__":
    #main()
    postprocess()
