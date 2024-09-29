#!/bin/bash
WORKSPACE_DIR=/data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/training/pt
CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")
LOG_FILE=${WORKSPACE_DIR}/${CURRENT_DATETIME}.log

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup deepspeed --num_gpus 4 /data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/src/train.py \
    --deepspeed /data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --stage pt \
    --do_train \
    --model_name_or_path /data/fcl/fcl/workspace/model/llama3_512_tokenizer \
    --dataset pt_512_c4_format \
    --dataset_dir /data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/data \
    --finetuning_type lora \
    --lora_target o_proj,v_proj,up_proj,k_proj,down_proj,q_proj,gate_proj \
    --additional_target linear_map,lm_head \
    --resize_vocab \
    --output_dir /data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/saves/llama3-8b_512_pretrain_lora_240925_1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --max_samples 1000000 \
    --preprocessing_num_workers 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --eval_steps 100 \
    --eval_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --ddp_timeout 1800000 \
    --plot_loss \
    --bf16 > ${LOG_FILE} 2>&1 &




    
# {
#     // 使用 IntelliSense 了解相关属性。 
#     // 悬停以查看现有属性的描述。
#     // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python 调试程序: 当前文件",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "${file}",
#             "console": "integratedTerminal"
#         },
#         {
#             "name": "deepspeed_z3",
#             "type": "python",
#             "request": "launch",
#             "program": "/data/fcl/anaconda3/envs/protein/bin/deepspeed", //"${file}",
#             "justMyCode": false,
#             "console": "integratedTerminal",
#             "args": [
#                 "--num_gpus", "4",
#                 "/data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/src/train.py",
#                 "--deepspeed", "/data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/examples/deepspeed/ds_z3_config.json",
#                 "--stage", "pt",
#                 "--do_train", "True",
#                 "--model_name_or_path", "/data/fcl/fcl/workspace/model/llama3_512_tokenizer",
#                 "--dataset", "pt_512_c4_format",
#                 "--dataset_dir", "/data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/data",
#                 "--finetuning_type", "lora",
#                 "--lora_target", "all",
#                 "--additional_target", "embed_tokens,lm_head",
#                 "--resize_vocab", "True",
#                 "--output_dir", "/data/fcl/fcl/workspace/2024_35/240724_protein/17_train_extent_llama3/LLaMA-Factory/saves/llama3-8b_512_pretrain_lora_debug_240925_1",
#                 "--overwrite_cache", "True",
#                 "--overwrite_output_dir", "True",
#                 "--cutoff_len", "1024",
#                 "--max_samples", "50",
#                 "--preprocessing_num_workers", "2",
#                 "--per_device_train_batch_size", "1",
#                 "--per_device_eval_batch_size", "1",
#                 "--gradient_accumulation_steps", "1",
#                 "--lr_scheduler_type", "cosine",
#                 "--logging_steps", "10",
#                 "--warmup_ratio", "0.1",
#                 "--save_steps", "100",
#                 "--eval_steps", "100",
#                 "--eval_strategy", "steps",
#                 "--learning_rate", "5e-5",
#                 "--num_train_epochs", "3.0",
#                 "--val_size", "0.1",
#                 "--ddp_timeout", "1800000",
#                 "--plot_loss", "True",
#                 "--bf16", "True",
#             ],
#             "env": {
#                 "CUDA_VISIBLE_DEVICES": "0,1,2,3",
#             }
#         },
#     ]
# }