#!/bin/bash

CONTEXT_LENGTH="context_short" # [context_short; context_long]
GOLDEN="dataset_${CONTEXT_LENGTH}.csv"
TEST_PREFIX="dataset_${CONTEXT_LENGTH}_eval"
OUTPUT_FILE="eval_metric_models_all.txt"

echo "csv_name,task_1,task_2,task_3,task_4,task_5,match" > ${OUTPUT_FILE} 

# google gemini models
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gemini-2.5-pro.csv"        >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gemini-2.5-flash.csv"      >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gemini-2.5-flash-lite.csv" >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gemini-2.0-flash.csv"      >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gemini-2.0-flash-lite.csv" >> ${OUTPUT_FILE} 

# openai gpt models
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-5.csv"               >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-5-mini.csv"          >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-5-nano.csv"          >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-4.1.csv"             >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-4.1-mini.csv"        >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-4.1-nano.csv"        >> ${OUTPUT_FILE} 
python eval_metric.py ${GOLDEN} "${TEST_PREFIX}_gpt-4o.csv"              >> ${OUTPUT_FILE} 
