# Regenerating the dataset 

This directory contains the code to regenerate the datset from scratch. The generated dataset, short and long context versions, can be found in [`dataset`](dataset/).

We perform two tasks:

- Task 1: Numerical Entity Extraction
- Task 2: Contextual Numerical Reasoning (multi-hop)

For prompts to generate question-answer pairs from sentences for task 1, and answers based on prompts for task 2, please see [`prompts.py`](generate_dataset/prompts.py).

## Example

Sentence: *7UP brands represented an estimated 3.9% of the total domestic soft drink market in 1993, down from 4.0% in 1992.*

`Task 1: Numerical Entity Extraction`

- Q: What was the 7UP brands market shrae in 1993?
-  3.9%

`Task 2: Contextual Numerical Reasoning plus multi-hop (maybe across sentences or sections)`

- Q1: Did 7UP’s market share increase or decrease compared to the previous year?
- A1: Decrease
- Q2: By how much did the 7UP brands market share decrease?
- A2: 0.1% (4.0% - 3.9%)

The sentence mentions a metric that decreased or increased from one year to the other year. 

a) Did the metric decrease or increase?

b) By how much did the metric decrease or increase? By how much did the metric change?

## How to run

### Install Python packages

Use the `requirements.txt` to install necessary Python packages. Any Python3 version should work, we used Python 3.12.1:

```
pip install -r requirements.txt
```

### Prerequisites

Check the config file [`config.json`](generate_dataset/config.json). For scripts that include Google or OpenAI API access, an **API key** must be specified. The Python scripts expect a json file named either `openai_api_key.json` or `google_api_key.json` in the directory `numerical_reasoning/generate_dataset` (and depending on the first keyword in the name i.e., `openai` or `google` uses separate API wrappers). This file must be valid json with the key `API_KEY`, for example

```
{
    "API_KEY": "MYAPIKEYABCDEFGHJKLMNOPQRSTUVXYZ"
}
```

### Steps to generate the dataset

1. Run [`edgar-crawler`](edgar-crawler) using `edgar-crawler/config.json` parameters as required. We focused on `ticker`s for the [`dow_jones.txt`](tickers/dow_jones.txt). Tickers are unique company identifiers on EDGAR. The list of Dow Jones tickers includes 30 companies. The remaining scripts are in the `generate_dataset` dir. 
2. [`split_sentences.py`](generate_dataset/split_sentences.py) using [`config.json`](generate_dataset/config.json) in the same dir. 
- Builds on spacy tagging, possible to enable parallel execution, see config. 
- The output dir is specified in the `config.json` and will include `.txt` files. 
- Each txt file is named similar to `354950_10K_2022_0000354950-22-000070_item_6.txt`. 
- Item is the section of the 10-K filing. Each txt file contains one sentence per line.
3. [`generate_questions.py`](generate_dataset/generate_questions.py): 
- From the sentences, we generate question-answer (QA)-pairs which make up a triplet including the sentence: (sentence, question, answer). 
- In the first run, we used `gpt-4o` to first generate questions from the sentence, with the question targeting the numeric value in the sentence. With the first run, not all samples could be processed as we ran out of credit on the OpenAI API. 
- In subsequent runs, we could use any model, e.g., Meta's Llama4 series or Google's Gemini models. 
- The output dir for `generate_questions.py` is specified in the `config.json`. 
- This output dir will include json files named `63908_10K_2006_0001193125-07-039707_item_7_91.json` where the name is similar to the original text file of the respective sentence - here now with the suffix `_91` which is a sentence id per item (unique per item and filing). 
- We perform *some* filtering for valid sentences and answers at this stage (see scripts).
4. [`filter_and_answer_tagging.py`](generate_dataset/filter_and_answer_tagging.py) 
- filters for weird/invalid criteria such as *sentence is too short* or *answer is too long*, *setence must start with capital letter followed by lowercase letter* and *sentence must end with a fullstop* (see scripts for more). 
- Furthermore, we perform spacy tagging on the generated answer from step 4: Based on the first run of `generate_questions.py` in step 4, we saw the answer labels `MONEY` and `PERCENT` worthwile including in the final dataset as the we can count the number of dollar and percent signs (`$` and `%`) in the corresponding sentences. 
- For instance, counting two dollar signs and two subsequent (unique) years, we have high chances to have found a sentence that goes like: *In addition, to enhance local relevance by having local franchisees operate more restaurants and to improve returns, we reduced our percentage of Company-operated restaurants in the U.K. from 63% at the end of 2005 to 54% at the end of 2006.*
5. `filter_and_answer_tagging_post_processing.py` accumulates and saves the data in one/two dataframes.
6. [`generate_task_2.py`](generate_dataset/generate_task_2.py) 
- runs `gemini` inference to generate answers for task 2. 
- We are interested in *quantitative* and *qualitative* answers i.e., 
- first *how did the financial metric change [PERCENT or MONEY]? answer qualitatively with 'increase' or 'decrease'*, 
- second *by how much did the financial metric change from the earlier year to the next year? answer quantitatively by stating the numeric difference*. 
- see [`prompts.py`](generate_dataset/prompts.py) for more.
7. [`generate_task_2_post_processing.py`](generate_dataset/generate_task_2_post_processing.py): 
- creates a dataframe and saves the filtered sentences and answers from task 1, with the results for the two categories of labelled task 1 answers (`MONEY` and `PERCENT`) to a csv. 
- This csv also includes the two answers for task 2 (quantitative and qualitative answers targeting the numeric change of the respective financial metric in the sentence).
8. [`generate_context_specific_questions.py`](generate_dataset/generate_context_specific_questions.py):
- generates questions based on the respective context, customising the question to the respective financial metric and report years extracted from the context
