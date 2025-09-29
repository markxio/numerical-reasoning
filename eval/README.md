# Evaluating models on our dataset

Provided scripts:
- [eval.py](eval.py): Calling the GPT and Gemini APIs using the Python package `openai`, generating answers for the tasks 1 to task 5 and saving the generated answers to file
- [eval_gemini.sh](eval_gemini.sh): Bash run script to test Gemini models and generate answers based on the dataset questions
- [eval_gpt.sh](eval_gpt.sh): Bash run script to test GPT models and generate answers based on the dataset questions
- [eval_metric.py](eval_metric.py): Comparing the generated model replies to the golden groundtruth, saving partial and exact match metrics to file
- [eval_metric.sh](eval_metric.sh): Bash run script evaluating model output to the golden groundtruth
