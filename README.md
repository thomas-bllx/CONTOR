# CONTOR

Models for QLoRa:
    - meta-llama/Llama-2-7b-hf
    - meta-llama/Llama-2-7b-chat-hf
    - togethercomputer/Llama-2-7B-32K-Instruct
    - meta-llama/Llama-2-13b-hf 
    - meta-llama/Llama-2-13b-chat-hf 
    - mistralai/Mistral-7B-v0.3 
    - mistralai/Mistral-7B-Instruct-v0.2 
    - lmsys/vicuna-13b-v1.5 
    - lmsys/vicuna-13b-v1.5-16k
    - meta-llama/Meta-Llama-3-8B
    - meta-llama/Meta-Llama-3-8B-Instruct

Models for unsloth:
    - unsloth/mistral-7b-v0.3-bnb-4bit
    - unsloth/mistral-7b-instruct-v0.3-bnb-4bit
    - unsloth/llama-3-8b-bnb-4bit
    - unsloth/llama-3-8b-Instruct-bnb-4bit
    - unsloth/Phi-3-medium-4k-instruct-bnb-4bit
    - unsloth/gemma-7b-bnb-4bit
    - unsloth/gemma-7b-it-bnb-4bit

Exemple of xp:
python3 main.py --theme all --model_name meta-llama/Llama-2-7b-hf --quantize qlora --prompt 0 --dataset untyped --splited 0 --output ./output
