from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
import os
import json
from tqdm import tqdm
import pandas as pd

from prompts import generate_multi_hop_qualitative, generate_multi_hop_quantitative
from generate_questions import get_unprocessed_files_only

cols_percent = [
    "json_file_path", 
    "year", 
    "ticker_id", 
    "item", 
    "sentence_id", 
    "answer_labels", 
    "sentence", 
    "question", 
    "answer", 
    "has_percent_sign_twice", 
    "has_percent_sign_thrice", 
    "has_two_years", 
    "has_three_years"
]

cols_money = [
    "json_file_path", 
    "year", 
    "ticker_id", 
    "item", 
    "sentence_id", 
    "answer_labels", 
    "sentence", 
    "question", 
    "answer", 
    "has_dollar_sign_twice", 
    "has_dollar_sign_thrice", 
    "has_two_years", 
    "has_three_years"
]

with open("google_api_key.json") as f:
    API_KEY = json.load(f)["API_KEY"]

def main():

    with open("config.json") as f:
        config = json.load(f)["generate_task_2"]

    print("-------------------------------")
    print("---------- config -------------")
    print("-------------------------------")
    pprint(config)
    print("-------------------------------")
    print("-------------------------------")
    print("-------------------------------")

    #########################
    # set config args starts
    #########################

    json_response_dir = config['json_response_dir']
    model_name = config['model_name']
    # TODO: implement check 
    check_already_processed = config['check_already_processed']

    #########################
    # set config args ends
    #########################
 
    os.system(f"rm -rf {json_response_dir}")
    os.system(f"mkdir -p {json_response_dir}/quantitative")
    os.system(f"mkdir -p {json_response_dir}/qualitative")

    # for now hardcoded the two labels MONEY and PERCENT
    # TODO: label list as config arg
    # currently hardcoded csv file
    # TODO: add csv input file as config arg
    money = pd.read_csv("FILTER_label_MONEY.csv", sep=";", usecols=cols_money)

    money2 = money[money["has_two_years"]==True]
    money2 = money2[money2["has_dollar_sign_twice"]==True]
    
    money3 = money[money["has_three_years"]==True]
    money3 = money3[money3["has_dollar_sign_thrice"]==True]

    money = pd.concat([money2, money3], ignore_index=True)

    percent = pd.read_csv("FILTER_label_PERCENT.csv", sep=";", usecols=cols_percent)

    percent2 = percent[percent["has_percent_sign_twice"]==True]
    percent2 = percent2[percent2["has_two_years"]==True]

    percent3 = percent[percent["has_percent_sign_thrice"]==True]
    percent3 = percent3[percent3["has_three_years"]==True]

    percent = pd.concat([percent2, percent3], ignore_index=True)

    dfs = {
        "money": money,
        "percent": percent,
    }
  
    dfs_output = {
        "money": dfs["money"].copy(),
        "percent": dfs["percent"].copy(),
    }

    system_instructions = {}
    system_instructions["qualitative"] = generate_multi_hop_qualitative
    system_instructions["quantitative"] = generate_multi_hop_quantitative

    for label, df in dfs.items(): 
        answers = {} 
        for json_file_path, sentence in zip(tqdm(df["json_file_path"].tolist()), df["sentence"].tolist()):
            for answer_type, sys_instruct in system_instructions.items():
                response = send_request_gemini(system_instruction=sys_instruct, user_content=sentence, config=config) 
                if answer_type not in answers:
                    answers[answer_type]=[]
                current_list = answers[answer_type]
                current_list.append(response["change"])
                answers[answer_type] = current_list

                # quickly write json to file
                filename=json_file_path.split("/")[-1]
                with open(f"{json_response_dir}/{answer_type}/{filename}", "w") as f:
                    json.dump(response, f)

        new_df = dfs_output[label].copy()
        for answer_type, sys_instruct in system_instructions.items():
            # write to file
            #if answer_type in answers:
            new_df[f"answer_2_{answer_type}"] = answers[answer_type] # only works for continuous indices
        new_df.to_csv(f"FILTER_label_{label}_task_2_answers_{model_name}.csv", sep=";", index=False)
 
class QuestionAnswer(BaseModel):
    change: str

def send_request_gemini(system_instruction, user_content, config):
    client = genai.Client(api_key=API_KEY)

    # https://ai.google.dev/gemini-api/docs/models 

    # 2.5 Pro
    # Our most powerful thinking model with maximum response accuracy
    # and state-of-the-art performance
    
    # 1000 requests per day maximum with my Tier1 key. 
    model_name=config['model_name']
    
    # 2.5 Flash
    # Our best model in terms of price-performance, 
    # offering well-rounded  capabilities.

    #model_name="gemini-2.5-flash-preview-04-17"
    #model_name="gemini-2.5-flash-preview-05-20" # new 

    response = client.models.generate_content(
        contents=user_content,
        model=model_name,
        config=GenerateContentConfig(
            system_instruction = system_instruction,
            response_mime_type = "application/json",
            response_schema = QuestionAnswer
        ),
        #config=types.GenerateContentConfig(
        #    max_output_tokens=10,
        #    temperature=0.5
        #)
    )

    return response.text

if __name__=="__main__":
    main()
