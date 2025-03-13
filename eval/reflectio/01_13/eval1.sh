cd ..
cd ..

bash run_eval.sh classification/gender_classification/classifier_no_vllm_phi_option_lora.py  --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_121927_balance/model_1784_balance/Phi-3.5-vision-instruct-lora

bash run_eval.sh classification/gender_classification/classifier_no_vllm_phi_lora.py  --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_121927_balance/model_1784_balance/Phi-3.5-vision-instruct-lora