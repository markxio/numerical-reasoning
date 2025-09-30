# llama 4 scout 17b 16e on 1x H200
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-4-Scout-17B-16E --port 8000 --max-model-len 32768 --max-num-batched-tokens 256000 --max-num-seqs 512 --gpu-memory-utilization 0.99 --enable-chunked-prefill --enable-prefix-caching --cpu-offload-gb 120
