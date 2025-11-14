import json
import re
import os
import subprocess
import tempfile
import ray
from ray.experimental.tqdm_ray import tqdm
import psutil
import argparse

class VerilogExecutionError(Exception):
    def __init__(self, message,error_type):
        super().__init__(error_type)
        self.type=error_type
        self.error_message=message

def kill_processes(active_processes):
    for process in active_processes:
        if process.poll() is None:  
            kill_process_tree(process.pid)
    active_processes.clear()  

def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):  
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass  
    except:
        pass


parser = argparse.ArgumentParser(description='Process some test.')

parser.add_argument('--rollout_data', type=str) 
parser.add_argument('--sim_info_path', type=str) 
parser.add_argument('--ast_save_path', type=str) 
args = parser.parse_args()

os.makedirs("/dev/shm/tmpdir_v", exist_ok=True)

@ray.remote(num_cpus=1)
def single_core_process(chunck_rollout_data,chunck_sim_info_data):
    results_ls=[]
    question_len=len(chunck_rollout_data)

    for i in tqdm(range(question_len),desc="questions",position=0):    
        response_candidates=chunck_rollout_data[i]
        sim_info_ls=chunck_sim_info_data[i]
        ast_ls=[]
        for response,sim_info in zip(response_candidates,sim_info_ls):
            if type(sim_info)==dict:
                pattern = r"```verilog\s*(.*?)\s*```"
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    code = matches[-1]
                    yosys_result=run_yosys(code)
                    if yosys_result[0]:
                        ast_ls.append(yosys_result[1])
                    else:
                        ast_ls.append("NULL")
                else:
                    ast_ls.append("NULL")
            else:
                ast_ls.append("NULL")
        results_ls.append(ast_ls)
    return results_ls

def run_yosys(code):
    active_processes = []
    
    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm/tmpdir_v") as tmpdir:
            v_path=os.path.join(tmpdir,"test.v")
            with open(v_path,"w") as f:
                f.write(code)

            iverilog_process = subprocess.Popen(
                f"yosys -p 'read_verilog -sv -dump_ast1 {v_path}'",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmpdir,shell=True
            )
            active_processes.append(iverilog_process) 

            stdout, stderr = iverilog_process.communicate(timeout=1)
            if iverilog_process.returncode != 0:
                raise VerilogExecutionError(stderr,"execute error")
            
            pattern="Dumping AST before simplification:(.*)--- END OF AST DUMP ---"
            match = re.search(pattern, stdout, re.DOTALL)
            if match:
                ast_text = match.group(1).strip()
            else:
                raise VerilogExecutionError("can not match AST","execute error")

            return True,ast_text
    except VerilogExecutionError as e:
        kill_processes(active_processes)
        return False,"execute error"
    except subprocess.TimeoutExpired:
        kill_processes(active_processes)
        return False,"timeout error"
    except:
        kill_processes(active_processes)
        return False,"other error"

if __name__ == "__main__":
    # Load Data
    sim_info_data=json.load(open(args.sim_info_path))
    rollout_data=json.load(open(args.rollout_data))
    print("Sim info data loaded", len(sim_info_data))
    
    ray.init()
    
    NUM_WORKERS = 80
    futures=[]

    chunck_size=len(sim_info_data)//NUM_WORKERS
    for i in range(NUM_WORKERS):
        if i!=NUM_WORKERS-1:
            chunck_sim_info_data=sim_info_data[i*chunck_size:(i+1)*chunck_size]
            chunck_rollout_data=rollout_data[i*chunck_size:(i+1)*chunck_size]
        else:
            chunck_sim_info_data=sim_info_data[i*chunck_size:]
            chunck_rollout_data=rollout_data[i*chunck_size:]
        futures.append(single_core_process.remote(chunck_rollout_data, chunck_sim_info_data))
    
    all_ast=[]
    for future in futures:
        result = ray.get(future)
        all_ast.extend(result)

    print("AST results len",len(all_ast))
    
    with open(args.ast_save_path, "w", encoding="utf-8") as f:
        json.dump(all_ast, f, indent=2, ensure_ascii=False)
    print(f"✅ saved to {args.ast_save_path}")

    ray.shutdown()

