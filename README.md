# QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation
![overview](./assets/overview.svg)

QiMeng-SALV introduces a novel framework for Verilog code generation that shifts reinforcement learning optimization from module-level to signal-level rewards. By leveraging AST analysis and signal-aware verification, it extracts functionally correct code segments from partially incorrect modules, enabling more effective RL training.

Resources:
- Webpage: https://zy1xxx.github.io/SALV/
- Paper: https://arxiv.org/abs/2510.19296
- Model: https://huggingface.co/TabCanNotTab/SALV-Qwen2.5-Coder-7B-Instruct
- Dataset: https://huggingface.co/datasets/TabCanNotTab/SALV-dataset

## Usage 
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "TabCanNotTab/SALV-Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
Please act as a professional verilog designer.
Implement a module of an 8-bit adder with multiple bit-level adders in combinational logic. 
Module name:  
    adder_8bit               
Input ports:
    a[7:0]: 8-bit input operand A.
    b[7:0]: 8-bit input operand B.
    cin: Carry-in input.
Output ports:
    sum[7:0]: 8-bit output representing the sum of A and B.
    cout: Carry-out output.
Implementation:
The module utilizes a series of bit-level adders (full adders) to perform the addition operation.
Give me the complete code.
"""

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# inference
outputs = model.generate(**model_inputs, max_new_tokens=2048, do_sample=True, temperature=0.5, top_p=0.95)

# get response text
input_length = model_inputs.input_ids.shape[1]
generated_tokens = outputs[0][input_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# get code text
pattern = r"```verilog\s*(.*?)\s*```"
matches = re.findall(pattern, response, re.DOTALL)
if matches:
    code=matches[-1]
    print(code)
else:
    print("No Verilog code found in the response!")
```

## Training
### Install
1. LLaMA-Factory for SALV
```
cd LLaMA-Factory
pip install -e .
```

2. iverilog and yoysy
```
sudo apt install iverilog
sudo apt install yosys
```

### SFT Training
1. Prepare training data
```
cd SFT
sh get_sft_training_data.sh
```
2. Train SFT model with LLaMA-Factory
```
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train ../SFT/train_sft.yaml
```

For more details, please refer to `SFT/README.md`.
### Signal-aware DPO
1. Prepare training data
```
cd SA-DPO
sh get_sa_dpo_training_data.sh
```

2. Train SA-DPO model with LLaMA-Factory
```
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train ../SA-DPO/train_sa_dpo.yaml
```

For more details, please refer to `SA-DPO/README.md`.

## Citation
```
@misc{zhang2025qimengsalvsignalawarelearningverilog,
  title={QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation}, 
  author={Yang Zhang and Rui Zhang and Jiaming Guo and Lei Huang and Di Huang and Yunpu Zhao and Shuyao Cheng and Pengwei Jin and Chongxiao Li and Zidong Du and Xing Hu and Qi Guo and Yunji Chen},
  year={2025},
  eprint={2510.19296},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.19296}, 
}
```

