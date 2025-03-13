cd ..
cd ..

PYTHONPATH=/net/papilio/storage7/tingyuan/llama/bias/vlm_bias python open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111101 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_235218_111101/model_checkpoint-1486_111101/Phi-3.5-vision-instruct-lora

PYTHONPATH=/net/papilio/storage7/tingyuan/llama/bias/vlm_bias python open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111110 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_235316_111110/model_checkpoint-1486_111110/Phi-3.5-vision-instruct-lora