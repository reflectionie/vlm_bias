cd ..
cd ..

PYTHONPATH=/net/papilio/storage7/tingyuan/llama/bias/vlm_bias python open_ended/image_tasks/story_generation/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111111 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_234514_111111/model_checkpoint-1486_111111/Phi-3.5-vision-instruct-lora

PYTHONPATH=/net/papilio/storage7/tingyuan/llama/bias/vlm_bias python open_ended/image_tasks/story_generation/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 011111 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_234731_011111/model_checkpoint-1486_011111/Phi-3.5-vision-instruct-lora