cd ..
cd ..

# image description 
PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm.py --custom_suffix original

PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111111 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_234514_111111/model_checkpoint-1486_111111/Phi-3.5-vision-instruct-lora

PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 011111 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_234731_011111/model_checkpoint-1486_011111/Phi-3.5-vision-instruct-lora

PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 101111 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_234836_101111/model_checkpoint-1486_101111/Phi-3.5-vision-instruct-lora

PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111101 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_235218_111101/model_checkpoint-1486_111101/Phi-3.5-vision-instruct-lora

PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111110 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_235316_111110/model_checkpoint-1486_111110/Phi-3.5-vision-instruct-lora

PYTHONPATH=. python eval/open_ended/image_tasks/image_description/Phi-3.5-vision-instruct/description_generation_no_vllm_lora.py --custom_suffix 111011 --model_path /net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_235900_111011/model_checkpoint-1486_111011/Phi-3.5-vision-instruct-lora