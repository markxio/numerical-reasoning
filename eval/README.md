# Evaluating models on our dataset

Generating model answers based on the five tasks and questions in the dataset:
- [eval.py](eval.py): Calling the GPT and Gemini APIs using the Python package `openai`, generating answers for the tasks 1 to task 5 and saving the generated answers to file
- [eval_gemini.sh](eval_gemini.sh): Bash run script to test Gemini models and generate answers based on the dataset questions
- [eval_gpt.sh](eval_gpt.sh): Bash run script to test GPT models and generate answers based on the dataset questions

Evaluating meta-llama models:
1. Deploy the `vllm/vllm-openai:latest` Docker image on Kubernetes using the provided [job-vllm.yaml](vllm/job-vllm.yaml). This enables us to run meta-llama (and other) models locally with vllm providing an `openai` compliant API endpoint on localhost. Ensure the model is locally available in case compute nodes are offline without internet connection.
2. Run vllm with the provided [serve_vllm.sh](vllm/serve_vllm.sh) run script.
3. [eval_llama.sh](eval_llama.sh): Bash run script, currently for `meta-llama/Llama-4-Scout-17B-16E`

Comparing and evaluating model answers to the golden groundtruth:
- [eval_metric.py](eval_metric.py): Comparing the generated model replies to the golden groundtruth, saving partial and exact match metrics to file
- [eval_metric.sh](eval_metric.sh): Bash run script evaluating model output to the golden groundtruth


