cd ..
cd ..

bash run_eval.sh classification/physical_classification/classifier_no_vllm_phi_option_class_level_acc_lora.py --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250113_192403_syoto_10111/model_checkpoint-1250_10111/Phi-3.5-vision-instruct-lora

bash run_eval.sh classification/physical_classification/classifier_no_vllm_phi_word_class_level_acc_lora.py --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250113_192403_syoto_10111/model_checkpoint-1250_10111/Phi-3.5-vision-instruct-lora