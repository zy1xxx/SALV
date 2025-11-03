import json 
import re
import argparse

parser = argparse.ArgumentParser(description='Process some test.')
parser.add_argument('--seed_data', type=str) 
parser.add_argument('--sim_info_path', type=str) 
parser.add_argument('--rollout_data', type=str) 
parser.add_argument('--training_data_save_path', type=str) 
args = parser.parse_args()

if __name__ == "__main__":
    # Load Data
    sim_info_data=json.load(open(args.sim_info_path))
    seed_data=json.load(open(args.seed_data))
    rollout_data=json.load(open(args.rollout_data))

    training_data=[]
    for idx,sim_info in enumerate(sim_info_data):
        instruction=seed_data[idx]["instruction"]
        gold_code=seed_data[idx]["gold_code"]

        success_flag=False
        for jdx,sim_item in enumerate(sim_info):
            if type(sim_item)==dict:
                rollout_response=rollout_data[idx][jdx]
                pattern = r"```verilog\s*(.*?)\s*```"
                matches = re.findall(pattern, rollout_response, re.DOTALL)
                if matches:
                    rollout_code = matches[-1]
                else:
                    continue

                # Check weather the test code is correct
                pass_flag=True
                for signa_name,error_ctn in sim_item.items():
                    if error_ctn!=0:
                        pass_flag=False
                
                # Add correct code into training data
                if pass_flag and len(sim_item)!=0:
                    training_data.append({"instruction":instruction,"response":rollout_response})
                    success_flag=True
        
        # Add all gold code into training data
        training_data.append({"instruction":instruction,"response":f"```verilog\n{gold_code}\n```"})
    
    print("SFT training data len:",len(training_data))    
    json.dump(training_data,open(args.training_data_save_path,"w"),indent=2)