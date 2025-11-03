# <img src="assets/logo.png" alt="Oracle" width="50"> Oracle: Investigating Advanced Reasoning of Large Language Models via Black-Box Interaction

---

<p align="center">
    <a href="https://arxiv.org/abs/2508.19035" target="_blank" rel="noopener noreferrer">
        <img alt="paper" src="https://img.shields.io/badge/paper-paper?logo=arxiv&logoColor=%23B31B1B&labelColor=white&color=%23B31B1B">
    </a>
    <a href="https://oraclebenchmark.github.io" target="_blank" rel="noopener noreferrer">
        <img alt="website" src="https://img.shields.io/badge/website-website?logo=safari&logoColor=%23006CFF&labelColor=white&color=%23006CFF">
    </a>
</p>

The Oracle benchmark investigates whether LLMs can independently explore unknown environment via black-box interaction. Current benchmark consists of 6 black-box tasks and 96 black-boxes, 51 of them are easy and 45 of them are hard. 19 popular LLMs are benchmarked.
Please check out the paper for more details, and this repo will detail how to run the evaluation.


## 💥 News

- **[2025.11.03]** 🚀 Added flexible API key management and local model support (Hugging Face, vLLM)
- **[2025.9.22]** 🐛 Fix a bug that caused incorrect GSI task results
- **[2025.8.27]** 🎯 Paper available in arxiv.
- **[2025.8.23]** 🎯 We release the code for Oracle Benchmark v1.0.

## 📖 Quick Start

First install the necessary packages with (using a virtual environment is recommended, tested with python 3.11.3):
```
conda create --name oracle python=3.11.3
conda activate oracle
pip install -r requirements.txt
```
Then open ```.env``` file and replace "xxx" with your own LLM API key.

We provide three shell files to help quickly reproduce the performance of GPT-4.1 in the Oracle benchmark. 

* If you want to start a baseline test, try
    ```
    sh run_script_baseline.sh
    ```
    
* If you want to evaluate the performance of GPT-4.1 in a black-box named 'simple_substitution' from Encryption Rule Inference (ERI) task, try
    ```
    sh run_script_simple_substitution.sh
    ```

* If you want to start a concurrent 10@1 evaluation of all 19 benchmarked models for the Encryption Rule Inference (ERI) task, try
    ```
    sh run_script_concurrent.sh
    ```

The interaction history will be save under ```./history```, and the results will be saved under ```./results```.

## 🚀 New Features: Flexible Model Management

Oracle Benchmark now supports flexible API key management and local model deployment!

### Check Available API Keys

You can check which API keys are configured and which model families are available:

```bash
python main.py --eva_model_family gpt --check_api_keys
```

This will show:
- Which API keys are configured and available
- Which model families you can use
- List of all available models

### Use Only Available Models

You no longer need to configure ALL API keys. The benchmark will only initialize API clients for the models you're actually using:

```bash
# Only need OpenAI API key to run this
python main.py --eva_model_family gpt --eva_model_name gpt-4o --task code
```

### Local Model Support

Oracle Benchmark now supports running evaluations with local open-source models!

#### Using Hugging Face Models

```bash
# Test with a local Llama model
python main.py --eva_model_family local --eva_model_name llama-2-7b \
  --local_model_path meta-llama/Llama-2-7b-chat-hf --task puzzle

# Test with a local Mistral model
python main.py --eva_model_family local --eva_model_name mistral-7b \
  --local_model_path mistralai/Mistral-7B-Instruct-v0.2 --task code

# Test with a local Qwen model
python main.py --eva_model_family local --eva_model_name qwen2-7b \
  --local_model_path Qwen/Qwen2-7B-Instruct --task encryption
```

**Note:** To use local models, you need to install additional dependencies:
```bash
pip install transformers torch accelerate
# Optional: for quantization support
pip install bitsandbytes
```

#### Using vLLM for Faster Inference

If you have a vLLM server running, you can connect to it:

```bash
# Start vLLM server (in another terminal)
vllm serve Qwen/Qwen2-7B-Instruct --port 8000

# Run benchmark with vLLM
python main.py --eva_model_family local --eva_model_name qwen2-7b \
  --vllm_base_url http://localhost:8000/v1 --task code
```

**Note:** To use vLLM, install it with:
```bash
pip install vllm
```

### Custom Model Configuration

You can define your own models in a YAML configuration file:

1. Create a custom configuration file (e.g., `my_models.yaml`):

```yaml
models:
  - name: my-llama-7b
    family: local
    type: local
    path: /path/to/my/llama-7b
    device: cuda
    load_in_4bit: true  # Use 4-bit quantization to save memory

  - name: my-mistral-vllm
    family: local
    type: vllm
    base_url: http://localhost:8000/v1
    path: mistralai/Mistral-7B-Instruct-v0.2

  - name: my-qwen-8bit
    family: local
    type: local
    path: Qwen/Qwen2-7B-Instruct
    load_in_8bit: true
```

2. Use the configuration file:

```bash
python main.py --model_config my_models.yaml \
  --eva_model_family local --eva_model_name my-llama-7b --task code
```

### Example: Testing Multiple Small Models

```bash
# Test different local models on the same task
for model in llama-2-7b mistral-7b qwen2-7b; do
  python main.py --eva_model_family local --eva_model_name $model \
    --local_model_path $model --task puzzle
done
```

### Backward Compatibility

All existing functionality remains unchanged. If you have all API keys configured, the benchmark works exactly as before:

```bash
# Traditional usage still works
python main.py --eva_model_family gpt --eva_model_name gpt-4.1 --task code
```




## 📖 Easy Scaling

The Oracle benchmark features easy scaling. You can easily build a new black-box with the following steps:

1. If you want to build a black-box for existing tasks:
    * Add a json file that describes the mapping rules of a black-box to corresponding task folder (e.g., ```./task/encryption/easy```).
    * run ```python main.py --eva_model_family --eva_model_name gpt-4.1 --task 'task_name'```.

2. If you want to build a new black-box task:
    * Build a task folder under ```./task```, write the task introduction and system prompt for models in ```task_intro``` and ```player_system_prompt``` respectively.
    * (Optional) Add related functions for black-box implementation in ```./ckpt.py``` if needed.
    * Build a task folder under ```./platforms```, write request and system prompt for the automatic black-box generation framework.
    * Build a task folder under ```./test```, write prompt for test sample generator, and complete related code in ```TestSamplesGenerator``` class in ```./auto_generation.py```.
    * run ```python main.py --eva_model_family --eva_model_name gpt-4.1 --task 'task_name'```.

The generated interaction history during iterative debugging will be saved in ```./logs```.

## 🏆 Leaderboard
<img width="1354" alt="benchmark_overview" src="assets/benchmark10@1.png">

<img width="1354" alt="benchmark_overview" src="assets/benchmark20@2.png">

## ✨ Contribution
Congchi Yin, Tianyi Wu, Yankai Shu contribute to the code implementation. We welcome contributions of black-boxes and black-box tasks from the community.

## 📚 Citation

If you find our work useful, please cite us:
```
@misc{yin2025investigatingadvancedreasoninglarge,
      title={Investigating Advanced Reasoning of Large Language Models via Black-Box Interaction}, 
      author={Congchi Yin and Tianyi Wu and Yankai Shu and Alex Gu and Yunhan Wang and Jun Shao and Xun Jiang and Piji Li},
      year={2025},
      eprint={2508.19035},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.19035}, 
}
```