ln ../SFT/Data/codev_135k_seed_data.json Data/codev_135k_seed_data.json
# gen spec seed data
python gen_spec_seed_data.py --seed_data Data/codev_135k_seed_data.json --save_path Data/codev_135k_seed_data_spec.json
# rollout
python ../Utils/rollout.py --model_name ../SFT/Model/SALV-SFT --seed_data ../SFT/Data/codev_135k_seed_data.json --rollout_save_path Data/rollout_data.json --n 5 
python ../Utils/rollout.py --model_name ../SFT/Model/SALV-SFT --seed_data Data/codev_135k_seed_data_spec.json --rollout_save_path Data/rollout_data_spec.json --n 5 
# sim
python ../Utils/sim.py --seed_data Data/codev_135k_seed_data.json --rollout_data Data/rollout_data.json --sim_info_path Data/sim_info_data.json
python ../Utils/sim.py --seed_data Data/codev_135k_seed_data_spec.json --rollout_data Data/rollout_data_spec.json --sim_info_path Data/sim_info_data_spec.json
# gen ast text
python ./gen_AST.py --rollout_data Data/rollout_data.json --sim_info_path Data/sim_info_data.json --ast_save_path Data/ast_text.json
python ./gen_AST.py --rollout_data Data/rollout_data_spec.json --sim_info_path Data/sim_info_data_spec.json --ast_save_path Data/ast_text_spec.json
# gen sa-dpo training data
python ./gen_sa_dpo_training_data.py --model_name ../SFT/Model/SALV-SFT  --seed_data Data/codev_135k_seed_data.json --sim_info_path Data/sim_info_data.json --rollout_data Data/rollout_data.json --ast_path Data/ast_text.json --training_data_save_path Data/sa_dpo_training_data.json
python ./gen_sa_dpo_training_data.py --model_name ../SFT/Model/SALV-SFT  --seed_data Data/codev_135k_seed_data_spec.json --sim_info_path Data/sim_info_data_spec.json --rollout_data Data/rollout_data_spec.json --ast_path Data/ast_text_spec.json --training_data_save_path Data/sa_dpo_training_data_spec.json