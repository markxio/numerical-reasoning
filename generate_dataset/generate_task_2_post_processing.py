import pandas as pd
import json
import re
import collections

from generate_task_2 import cols_percent, cols_money
from split_sentences import pprint

def post_processing():
    csvs = [
        "FILTER_label_money_task_2_answers.csv",
        "FILTER_label_percent_task_2_answers.csv"
    ]
    
    new_cols = ["answer_2_qualitative", "answer_2_quantitative"]

    dfs = []

    DO_TRANSFORMATION=False

    # inplace transformation: 
    # loads csv as df, changes df and writes to the same csv
    for idx, csv in enumerate(csvs):
        dfs.append(pd.read_csv(csv, sep=";"))
        if DO_TRANSFORMATION:
            for new_col in new_cols:
                json_series = dfs[idx][new_col].apply(json.loads)
                str_series = json_series.apply(lambda x: x["change"])
                dfs[idx][new_col] = str_series 

            dfs[idx].to_csv(csv, sep=";", index=False)
    
    df = pd.concat([dfs[0], dfs[1]], ignore_index=True)
    df = df.drop(columns=["has_percent_sign_twice", "has_percent_sign_thrice", "has_dollar_sign_twice", "has_dollar_sign_thrice"]) 
    df.to_csv("FILTER_labels_MONEY_PERCENT_task_2_answers.csv", sep=";", index=False)

def create_dataset_from_manual_review():
    df = pd.read_excel("FILTER_labels_MONEY_PERCENT_task_2_answers_dataset_exclude.xlsx", header=0, sheet_name="dataset")
    cols = ["json_file_path","year","ticker_id","item","sentence_id","answer_labels","sentence","task_1_question_gpt-4o","task_1_answer_gpt-4o","task_2_answer_qualitative_gemini-2.5-pro-preview-05-06","task_2_answer_quantitative_gemini-2.5-pro-preview-05-06","exclude"]

    df = df[cols]
    new_names = {
        "task_1_question_gpt-4o": "task_1_question",
        "task_1_answer_gpt-4o": "task_1_answer",
        "task_2_answer_qualitative_gemini-2.5-pro-preview-05-06": "task_2_answer_qualitative",
        "task_2_answer_quantitative_gemini-2.5-pro-preview-05-06": "task_2_answer_quantitative"
    }
    df = df.rename(columns=new_names)
    df_exclude = df[df["exclude"]=="YES"]
    df_new = df[df["exclude"]!="YES"]
    
    print(f"Before: {len(df.index)}")
    print(f"Exclude: {len(df_exclude.index)}")
    print(f"After: {len(df_new.index)}")

    df_new = df_new.drop(columns=["exclude"])
    df_new.to_csv("dataset.csv", index=False, sep=";")

if __name__=="__main__":
    #post_processing()
    create_dataset_from_manual_review()
