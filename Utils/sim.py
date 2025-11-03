import json
import re
import os
import subprocess
import tempfile
import ray
from ray.experimental.tqdm_ray import tqdm
import argparse
from utils import VerilogExecutionError, kill_process_tree

parser = argparse.ArgumentParser(description='Process some test.')
parser.add_argument('--seed_data', type=str) 
parser.add_argument('--rollout_data', type=str) 
parser.add_argument('--sim_info_path', type=str) 
args = parser.parse_args()

os.makedirs("/dev/shm/tmpdir_v", exist_ok=True)

def sim_testcode(predict: str, testbench: str) -> float:
    active_processes = []
    
    # Get test_code
    try:
        pattern = r"```verilog\s*(.*?)\s*```"
        matches = re.findall(pattern, predict, re.DOTALL)
        if matches:
            test_code = matches[-1]
            pattern = r"module\s+([a-zA-Z0-9_]+)\s*(#\s*\([\s\S]+\))?\s*(//.*\s)?\([\s\S]+?\);"
            match = re.search(pattern, test_code)
            if match:
                module_name = match.group(1)
                test_code = test_code.replace(module_name, f"{module_name}_test")
            else:
                return "no module head"
        else:
            return "no test code"
    except:
        return "no module head"
    
    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm/tmpdir_v") as tmpdir:
            # Create linked file
            v_path=os.path.join(tmpdir,"test.v")
            vvp_path=os.path.join(tmpdir,"test.vvp")
            with open(v_path,"w") as f:
                f.write(f"{test_code}\n{testbench}")
            
            # Compile
            iverilog_process = subprocess.Popen(
                f'iverilog -g2012 -o {vvp_path} {v_path}',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmpdir,shell=True
            )
            active_processes.append(iverilog_process)  

            stdout, stderr = iverilog_process.communicate(timeout=1)
            if iverilog_process.returncode != 0:
                raise VerilogExecutionError(stderr,"compile error")
            
            # Execute
            iverilog_process = subprocess.Popen(
                f'vvp {vvp_path}',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmpdir,shell=True
            )
            active_processes.append(iverilog_process) 

            stdout, stderr = iverilog_process.communicate(timeout=1)
            if iverilog_process.returncode != 0:
                raise VerilogExecutionError(stderr,"execute error")

            # Get execute result
            pattern = r"(.*)\serror ctn:\s(.*)" 
            match = re.findall(pattern, stdout)
            result={}
            if match:
                for item in match:
                    signal_name=item[0]
                    error_num=int(item[1])
                    result[signal_name]=error_num
                return result
            else:
                return "not match result"
    # Exception Process
    except VerilogExecutionError as e:
        if e.type == "compile error":
            return "compile error"
        elif e.type == "execute error":
            return "execute error"
        else:
            return "execute error"
    except subprocess.TimeoutExpired:
        return "timeout error"
    except:
        return "other sim error"
    finally:
        for process in active_processes:
            if process.poll() is None:  
                kill_process_tree(process.pid)
        active_processes.clear()  
    

@ray.remote(num_cpus=1)
def single_core_process(chunck_seed_data,chunck_rollout_data):
    results_ls=[]
    seed_data_len=len(chunck_seed_data)
    for i in tqdm(range(seed_data_len),desc="questions",position=0):    
        response_candidates=chunck_rollout_data[i]
        testbench=chunck_seed_data[i]["testbench"]+"\n"+chunck_seed_data[i]["gold_code"]
        results=[]
        for response in response_candidates:
            result=sim_testcode(response,testbench)
            results.append(result)
        results_ls.append(results)
    return results_ls

if __name__ == "__main__":
    ray.init()

    # Load Data
    seed_data=json.load(open(args.seed_data))
    rollout_data=json.load(open(args.rollout_data))
    print("Data loaded:", len(seed_data))

    NUM_WORKERS = 80
    futures=[]
    chunck_size=len(seed_data)//NUM_WORKERS
    for i in range(NUM_WORKERS):
        if i!=NUM_WORKERS-1:
            chunck_seed_data=seed_data[i*chunck_size:(i+1)*chunck_size]
            chunck_rollout_data=rollout_data[i*chunck_size:(i+1)*chunck_size]
        else:
            chunck_seed_data=seed_data[i*chunck_size:]
            chunck_rollout_data=rollout_data[i*chunck_size:]
        futures.append(single_core_process.remote(chunck_seed_data, chunck_rollout_data))
            
    # Collect results    
    all_results=[]
    for future in futures:
        result = ray.get(future)
        all_results.extend(result)
    print("Results len:", len(all_results))
    
    with open(args.sim_info_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ saved to {args.sim_info_path}")

    ray.shutdown()

