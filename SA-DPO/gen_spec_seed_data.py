import json 
import re
import argparse

parser = argparse.ArgumentParser(description='Process some test.')

parser.add_argument('--seed_data', type=str) 
parser.add_argument('--save_path', type=str) 
args = parser.parse_args()

problem_ls=json.load(open(args.seed_data,"r"))
new_ls=[]
for problem_item in problem_ls:
    problem_id=problem_item["problem_id"]
    instruction=problem_item["instruction"]
    gold_code=problem_item["gold_code"]
    port_names=problem_item["port_names"]
    testbench=problem_item["testbench"]
    pattern_module = r"module\s+([a-zA-Z0-9_]+)\s*(#\s*\([\s\S]+\))?\s*(//.*\s)?\([\s\S]+?\);"
    match = re.search(pattern_module, instruction, re.DOTALL)
    if match:
        module_group=match.group(0)
        module_name=match.group(1)
        description=instruction.replace(module_group,"")
        input_string=""
        output_string=""
        in_output_string=""
        for item in port_names:
            name=item["name"]
            if item["type"]=='input':
                input_string+=f"\t{name}\n"
            elif item["type"]=='output':
                output_string+=f"\t{name}\n"
            else:
                in_output_string+=f"\t{name}\n"
        prompt=f"Please act as a professional verilog designer.\n{description}\nModule name:\n\t{module_name}\nInput ports:\n{input_string}Output ports:\n{output_string}\nGive me the complete code."
        new_ls.append({"problem_id":problem_id,"instruction":prompt,"gold_code":gold_code,"port_names":port_names,"testbench":testbench})
json.dump(new_ls,open(args.save_path,"w"),indent=2)
