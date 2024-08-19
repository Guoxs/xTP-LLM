port=29500
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