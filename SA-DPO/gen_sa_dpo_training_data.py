import json
import re
from itertools import combinations
from parser import ASTTree
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import ray
import argparse

parser = argparse.ArgumentParser(description='Process some test.')

parser.add_argument('--model_name', type=str) 
parser.add_argument('--seed_data', type=str) 
parser.add_argument('--rollout_data', type=str) 
parser.add_argument('--sim_info_path', type=str) 
parser.add_argument('--ast_path', type=str)
parser.add_argument('--training_data_save_path', type=str) 
args = parser.parse_args()

def get_select_index(response, ast_text, signal_info):
    pattern = r"```verilog\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        code = matches[-1]
    else:
        raise ValueError

    ast_tree = ASTTree(ast_text, code)
    token_indexes = ast_tree.get_select_token_index(tokenizer, response, signal_info)
    return token_indexes

def check_complex(response):
    pattern = r'\bgenerate\b.*?\bendgenerate\b'  
    if re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE):
        return True 
    
    if re.search(r'\bfunction\b[\s\S]*?\bendfunction\b', response, re.IGNORECASE):
        return True 

    for key_word in ["interface","package"]:
        if key_word in response:
            return True
    
    return False

@ray.remote
def process_idx(sim_info, instruction, response_ls, ast_text_ls):
    pair_data_local = []
    new_sim_info = [(i, s) for i, s in enumerate(sim_info)]

    for sim_info_comb in combinations(new_sim_info, 2):
        sample_types = []
        responses = []
        ast_texts = []

        for i, s_info in sim_info_comb:
            responses.append(response_ls[i])
            ast_texts.append(ast_text_ls[i])
            if isinstance(s_info, dict):
                signal_result_ls = []
                for signa_name, error_ctn in s_info.items():
                    if error_ctn / 1000 == 0:
                        signal_result_ls.append("correct")
                    elif error_ctn/1000 > 0:
                        signal_result_ls.append("incorrect")
                    else:
                        signal_result_ls.append("ambiguous")
                    
                sample_types.append(signal_result_ls)
            else:
                pattern = r"```verilog\s*(.*?)\s*```"
                matches = re.findall(pattern, response_ls[i], re.DOTALL)
                if not matches:
                    sample_types.append("rejected")
                else:
                    sample_types.append("unknown")

        if "unknown" in sample_types:
            continue
        else:
            if isinstance(sample_types[0], list) and isinstance(sample_types[1], list):
                signal_names = sim_info_comb[0][1].keys()

                def add_training_data(index_A, index_B):
                    signal_info = {"correct_signals": [], "incorrect_signals": []}
                    for A_signal, B_signal, signal_name in zip(
                        sample_types[index_A], sample_types[index_B], signal_names
                    ):
                        if A_signal == "correct" and B_signal == "incorrect":
                            signal_info["correct_signals"].append(signal_name)

                    if len(signal_info["correct_signals"]) != 0:
                        signal_info["incorrect_signals"] = list(
                            set(signal_names) - set(signal_info["correct_signals"])
                        )
                        if (
                            ast_texts[index_A] != "NULL"
                            and ast_texts[index_B] != "NULL"
                        ):
                            if check_complex(responses[index_A]):
                                chosen_text=responses[index_A]
                            else:
                                chosen_text = responses[index_A] + "<split_token>" + json.dumps(
                                    get_select_index(responses[index_A], ast_texts[index_A], signal_info)
                                )
                            
                            if check_complex(responses[index_B]):
                                rejected_text=responses[index_B]
                            else:
                                rejected_text = responses[index_B] + "<split_token>" + json.dumps(
                                    get_select_index(responses[index_B], ast_texts[index_B], signal_info)
                                )
                            
                            pair_data_local.append(
                                {"instruction": instruction, "chosen": chosen_text, "rejected": rejected_text}
                            )

                add_training_data(0, 1)
                add_training_data(1, 0)
            
            elif isinstance(sample_types[0], str) or isinstance(sample_types[1], str):
                def add_training_data(index_A, index_B):
                    if (
                        sample_types[index_A] == "rejected"
                        and isinstance(sample_types[index_B], list)
                        and sample_types[index_B].count("correct") == len(sample_types[index_B])
                    ):
                        pair_data_local.append(
                            {"instruction": instruction, "chosen": responses[index_B], "rejected": responses[index_A]}
                        )

                add_training_data(0, 1)
                add_training_data(1, 0)

    return pair_data_local

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load Data
    start = time.time()
    seed_data = json.load(open(args.seed_data))
    rollout_data = json.load(open(args.rollout_data))
    sim_info_data = json.load(open(args.sim_info_path))
    ast_text_data = json.load(open(args.ast_path))
    
    print("Data Loaded, time cost:", time.time() - start)
    
    futures = [
        process_idx.remote(
            sim_info_data[idx],
            seed_data[idx]["instruction"],
            rollout_data[idx],
            ast_text_data[idx],
        )
        for idx in range(len(sim_info_data))
    ]

    pair_data = []
    for result in tqdm(ray.get(futures), total=len(futures)):
        pair_data.extend(result)

    print(len(pair_data))
    json.dump(pair_data, open(args.training_data_save_path, "w"), indent=2)
