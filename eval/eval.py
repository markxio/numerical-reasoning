#!/bin/python

import json
from tqdm import tqdm
import pandas as pd
import ast
import argparse

from pydantic import BaseModel
import openai
from openai import OpenAI
import time

class PromptsEval:
    def q1():
        return "What is the financial metric mentioned in the sentence? Name the financial metric as provided in the sentence."
    
    def q2():
        return "For which years is the financial metric reported? State the years in a comma-separated list"
    
    def q3(financial_metric: str, years: list[int]):
        return f"What was the value of the Company's {financial_metric} in {years[0]}? Only state the value as provided in the sentence."
    
    def q4(financial_metric: str, years: list[int]):
        return f"Did {financial_metric} from {years[0]} to {years[1]} change? Answer with 'increase', 'decrease' or 'constant'."

    def q5(financial_metric: str, years: list[int]):
        return f"By how much did {financial_metric} change from {years[0]} to {years[1]}? Only state the numeric difference as provided in the sentence."

def eval(long_context, vendor, api_key, model_name):
    print(f"Evaluating model {model_name} from vendor {vendor} ...")
    slug = "_context_short"
    if long_context:
        slug = "_context_long"

    dataset = pd.read_csv(f"dataset{slug}.csv", sep=";")

    answers = []
    answers.append([])
    answers.append([])
    answers.append([])

    if long_context == False:
        answers.append([])
        answers.append([])

    answers[0] = []
    answers[1] = []
    answers[2] = []

    if long_context == False:
        answers[3] = []
        answers[4] = []

    # need to escape any context due to json incompatibilities
    for context, metric, years in zip(tqdm(dataset["context"]), dataset["task_1_answer"], dataset["task_2_answer"]):    
        years_list = ast.literal_eval(years)
        questions = []
        if long_context == False:
            questions.append(PromptsEval.q1())
            questions.append(PromptsEval.q2())
        questions.append(PromptsEval.q3(metric, years_list))
        questions.append(PromptsEval.q4(metric, years_list))
        questions.append(PromptsEval.q5(metric, years_list))

        context_escaped = context.replace("\n", ", ").replace('"', '').replace("\r", ", ").replace(";", ":").replace("  ", " ").replace("\\", "\\\\").replace("\"", "\\\"").strip()

        for i, q in enumerate(questions):
            instruct = f"The following context stems from a SEC 10-K annual filing of a company. Answer the following question based on the given context."
            user_content = f"Context: {context_escaped}\n Question: {q}"
            response = send_request(system_content=instruct, user_content=user_content, vendor=vendor, api_key=api_key, model_name=model_name, response_format=Answer)
            #print(response)
            answer = response.answer
            answers[i].append(answer)

    if long_context:
        # long context is evaluated only on tasks 3 to 5
        dataset["task_3_answer"] = answers[0]
        dataset["task_4_answer"] = answers[1]
        dataset["task_5_answer"] = answers[2]
    else:
        dataset["task_1_answer"] = answers[0]
        dataset["task_2_answer"] = answers[1]
        dataset["task_3_answer"] = answers[2]
        dataset["task_4_answer"] = answers[3]
        dataset["task_5_answer"] = answers[4]

    dataset.to_csv(f"dataset{slug}_eval_{model_name.lower().replace('/', '-')}.csv", sep=";", index=False)
    
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

def send_request(system_content, user_content, vendor, api_key, model_name, response_format):

    # openai_api_key.json
    # gpt-4o

    # google_api_key.json
    # gemini-2.5-pro-preview-05-06

    if "google" in vendor:
        # use google gemini with openai python lib:
        # https://ai.google.dev/gemini-api/docs/openai
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        while True:
            try:
                #completion = client.chat.completions.create( # google docs recommend this
                completion = client.beta.chat.completions.parse( # uses response_format
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=QuestionAnswer
                )
                break
            except openai.RateLimitError as e:
                print("Rate limit hit. Waiting 30s before retrying...")
                time.sleep(30)  # wait before retrying
            except Exception as e:
                print("Other error:", e)
                print(completion.choices[0].message)
                raise
    elif "openai" in vendor:
        client = OpenAI(api_key=api_key)
        while True:
            try:
                #completion = client.chat.completions.create( 
                completion = client.beta.chat.completions.parse( # uses response_format
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=response_format #QuestionAnswer
                )
                break
            except openai.RateLimitError as e:
                print("Rate limit hit. Waiting 30s before retrying...")
                time.sleep(30)  # wait before retrying
            except Exception as e:
                print("Other error:", e)
                print(completion.choices[0].message)
                raise            
    elif "meta" in vendor:
        client = OpenAI(
            api_key=api_key,
            base_url="http://localhost:8000/v1"
        )
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=response_format #QuestionAnswer
        )
    else:
        raise ValueError("The API provider/vendor couldnt be identified (We can use the OpenAI python library to address the Google Gemini API).")

    return completion.choices[0].message.parsed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_names", nargs="+", type=str, help="List of model names")
    # default False (i.e. default is short context)
    parser.add_argument('--long-context', action='store_true')

    args = parser.parse_args()
    model_names = args.model_names
    long_context = args.long_context

    print(f"Model names: {model_names}")
    print(f"Long context: {long_context}")

    api_key_files = {
        "google": "google_api_key.json",
        "openai": "openai_api_key.json"
    }

    with open(api_key_files["google"]) as f:
        google = json.load(f)["API_KEY"]

    with open(api_key_files["openai"]) as f:
        openai = json.load(f)["API_KEY"] 

    api_keys = {}
    api_keys["google"] = google
    api_keys["openai"] = openai

    for model_name in model_names:
        vendor=""
        if "gpt" in model_name:
            vendor="openai"
            api_key = api_keys["openai"]
        elif "gemini" in model_name:
            vendor="google"
            api_key = api_keys["google"]
        elif "meta" in model_name:
            vendor="meta"
            api_key = "none"
        else:
            raise ValueError("The model name couldnt be identified (We can use the OpenAI python library to address the Google Gemini API).")
        eval(long_context, vendor, api_key, model_name)


if __name__=="__main__":
    main()
