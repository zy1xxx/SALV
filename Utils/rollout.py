import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from ray.experimental.tqdm_ray import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Process some test.')
parser.add_argument('--model_name', type=str,default="Qwen/Qwen2.5-Coder-7B-Instruct") 
parser.add_argument('--n', type=int) 
parser.add_argument('--seed_data', type=str) 
parser.add_argument('--rollout_save_path', type=str) 

args = parser.parse_args()

MINI_BATCH_P=2000
NUM_WORKERS = 8

@ray.remote(num_gpus=1)
class BatchedLLMWorker:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def rollout_batch(self, prompts: list[str], n: int = 5):
        batch = []
        for p in prompts:
            messages = [
                {"role": "system", "content": "You are a professional verilog designer. Give the complete verilog code"},
                {"role": "user", "content": p}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch.extend([text] * n)

        sampling_params = SamplingParams(
            temperature=1,
            top_p=0.95,
            max_tokens=2048,
        )
        
        mini_batch_size=MINI_BATCH_P*n
        steps=len(batch)//mini_batch_size if len(batch)%mini_batch_size==0 else len(batch)//mini_batch_size+1
        
        result = []
        for step in tqdm(range(steps), desc="steps", position=0):
            if (step+1)*mini_batch_size<len(batch):
                mini_batch=batch[step*mini_batch_size:(step+1)*mini_batch_size]
            else:
                mini_batch=batch[step*mini_batch_size:]
            outputs = self.llm.generate(mini_batch, sampling_params)
            
            for i in range(len(outputs)//n):
                responses = [outputs[i * n + j].outputs[0].text.strip() for j in range(n)]
                result.append(responses)
        return result

if __name__ == "__main__":
    # Load Prompts
    prompts=[]
    for item in json.load(open(args.seed_data)):
        prompts.append(item["instruction"])
    print("Sample problems len:", len(prompts))

    # init ray
    ray.init()

    workers = [BatchedLLMWorker.remote(args.model_name) for _ in range(NUM_WORKERS)]
    n = args.n

    futures=[]
    chunck_size=len(prompts)//NUM_WORKERS
    for i in range(NUM_WORKERS):
        if i != NUM_WORKERS-1:
            chunck_prompts=prompts[i*chunck_size:(i+1)*chunck_size]
        else:
            chunck_prompts=prompts[i*chunck_size:]
        worker = workers[i]  
        futures.append(worker.rollout_batch.remote(chunck_prompts, n=n))

    all_rollouts=[]
    for future in futures:
        result = ray.get(future)
        all_rollouts.extend(result)

    print("Results len:",len(all_rollouts))

    # Save Rollout Response
    output_path = os.path.join(args.rollout_save_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_rollouts, f, indent=2, ensure_ascii=False)
    print(f"✅ saved to {output_path}")

    ray.shutdown()
