port=29500
train_localhost='0,1,2,3'
test_localhost='0,1,2'

prefix='all'
his_len=12
pred_len=12
dir_name='mini_datasets'
base_model_path=./Llama_models/7B-chat/hf

suffix_str=${prefix}_${his_len}pred${pred_len}

model_name='Llama-2-7B-fp16'
setting_str='lora_r16_alpha32'
trial_name=${model_name}-traffic-prediction_${suffix_str}-${setting_str}
output_model=../outputs/${dir_name}/${trial_name}

# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir -p ${output_model}
fi

cp ./finetune_lora_chat.sh ${output_model}
deepspeed --master_port ${port} --include localhost:${train_localhost} finetune_lora.py \
    --model_name_or_path ${base_model_path} \
    --train_files ./${dir_name}/traffic_datasets_${suffix_str}_train_mini.json \
    --validation_files  ./${dir_name}/traffic_datasets_${suffix_str}_val_mini.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --experiment_id ${trial_name} \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 1000 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --warmup_steps 400 \
    --load_in_bits 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 64 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 50 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to wandb \
    --project_name R2T-LLM \
    --run_name ${trial_name} \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log 

    # --resume_from_checkpoint ${output_model}/checkpoint-350 \


deepspeed --master_port ${port} --include localhost:${test_localhost} inference_chat.py \
    --base_model_path ${base_model_path} \
    --validation_files  ./${dir_name}/traffic_datasets_${suffix_str}_test_mini.json \
    --new_model_path ${output_model} \
    --output_path ${output_model} \
    --max_new_tokens 100 \
    --pred_len ${pred_len} \
    --batch_size_each_gpu 8 \
    --infer_times 3 \
    --use_finetune_model 1