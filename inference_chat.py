import os
import re
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed
)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, hugg_datasets):
        self.hugg_datasets = hugg_datasets
        
    def __len__(self):
        return len(self.hugg_datasets)
    
    def __getitem__(self, i):
        input_pattern = r"\[INST\](.*?)\[/INST\]"
        prompt_input = re.findall(input_pattern, self.hugg_datasets[i]['text'], re.DOTALL)[0]
        return prompt_input


def infer_for_each_rank(model_pipe, dataset_pipe, local_rank, batch_size, pred_len, output_path):
    print("Evaluation times: ", local_rank)
    preds = []
    for results in tqdm(model_pipe(dataset_pipe, batch_size=batch_size), total=len(dataset_pipe)):
        outs = []
        for result in results:
            #print(results)
            #out = re.findall(r'\d+', re.search(r': \[(.*)\]', result['generated_text']).group())
            out = re.findall(r"(\d+)", result['generated_text'])[1:]
            #print(out)
            if len(out) == 0:
                pad = [np.inf] if (pred_len == 1) else ([np.inf] * pred_len)
                outs.append(pad)
                continue
            elif len(out) < pred_len:
                out += [np.inf] * (pred_len - len(out))
            elif len(out) > pred_len:
                out = out[:pred_len]
            out = [float(i) for i in out]
            #print(out)
            outs.append(out)
        preds += outs
    # save
    np.savetxt(os.path.join(output_path, 'preds_' + str(local_rank) + '.txt'), preds, fmt='%.1f')

    
def get_labels(hugg_data_val, pred_len, output_path):
    label_pattern = r"\[/INST\]{Traffic volume data in the next \d+ hours: \[(.*)\]}.</s>"
    labels = [re.findall(label_pattern, data['text'])[0].split(',') for data in hugg_data_val]
    labels = np.asarray(labels, dtype=np.int32)
    np.savetxt(os.path.join(output_path, 'labels.txt'), labels, fmt='%.1f')


def load_and_merge_preds(output_path, pred_times):
    all_preds = []
    for i in range(pred_times):
        preds_path = os.path.join(output_path, 'preds_' + str(i) + '.txt')
        preds = np.loadtxt(preds_path)
        all_preds.append(preds)

    all_preds = np.asarray(all_preds)
    if len(all_preds.shape) == 2:
        all_preds = all_preds[:, :, np.newaxis]
    
    # merge multiply preds to one
    all_preds_resized = np.transpose(all_preds, (1, 2, 0))
    pred_shape = all_preds_resized.shape # [N, pred_len, times]

    merged_preds = []
    for i in range(pred_shape[0]):
        tmp_pred = []
        for j in range(pred_shape[1]):
            item = all_preds_resized[i][j]
            no_inf_ids = np.where(~np.isinf(item))[0]
            valid_data = item[no_inf_ids]
            if len(valid_data) == 0:
                tmp_pred.append(np.inf)
            else:
                tmp_pred.append(np.mean(valid_data))
        if len(tmp_pred) == 1:
            tmp_pred = tmp_pred[0]
        merged_preds.append(tmp_pred)

    merged_preds = np.asarray(merged_preds)
    return merged_preds


def eval(labels, preds, output_path, infer_times):
    # remove sample that is np.inf in preds
    if type(labels[0]) == np.ndarray:
        no_inf_ids = np.where(~np.isinf(preds).any(axis=1))[0].tolist()
    else:
        no_inf_ids = np.where(~np.isinf(preds))[0].tolist()
        
    print("No inf ids in preds: ", len(no_inf_ids))
    no_inf_preds = preds[no_inf_ids]
    no_inf_labels = labels[no_inf_ids]

    # remove sample that is large that 2000
    if type(labels[0]) == np.ndarray:
        reasonable_ids = np.where((no_inf_preds < 1000).all(axis=1))[0].tolist()
    else:
        reasonable_ids = np.where(no_inf_preds < 1000)[0].tolist()
    
    print("Reasonable value ids in preds: ", len(reasonable_ids))
    all_refined_preds = no_inf_preds[reasonable_ids]
    all_refined_labels = no_inf_labels[reasonable_ids]

    rmses, maes, mapes = [], [], []
    for i in range(all_refined_labels.shape[1]):
        refined_preds = all_refined_preds[:, i]
        refined_labels = all_refined_labels[:, i]
        
        diff = refined_preds - refined_labels

        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))

        if (type(refined_preds[0]) == list) or (type(refined_preds[0]) == np.ndarray):
            no_zero_ids = np.where(np.all(refined_labels != 0, axis=1))[0].tolist()
            no_zero_ids = np.unique(no_zero_ids)
        else:
            no_zero_ids = np.where(refined_labels != 0)[0].tolist()

        print("No zero ids in labels: ", len(no_zero_ids))

        mape = np.mean(np.abs(diff[no_zero_ids] / refined_labels[no_zero_ids])) * 100

        print(f'RMSE: {rmse: .3f}')
        print(f'MAE: {mae: .3f}')
        print(f'MAPE: {mape: .3f}')
        
        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)

        file = open(os.path.join(output_path, f'horizion_{i+1}_times_{infer_times}_RMSE_{rmse:.2f}-MAE_{mae:.2f}-MAPE_{mape:.2f}'),'w')
        file.close()
   
    file = open(os.path.join(output_path, f'horizion_mean_times_{infer_times}_RMSE_{np.mean(rmses):.2f}-MAE_{np.mean(maes):.2f}-MAPE_{np.mean(mapes):.2f}'),'w')
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--base_model_path", type=str, default="Llama-2-7B-fp16")
    parser.add_argument("--validation_files", type=str, default="")
    parser.add_argument("--new_model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size_each_gpu", type=int, default=8)
    parser.add_argument("--infer_times", type=int, default=5)
    parser.add_argument("--use_finetune_model", type=int, default=1)

    args = parser.parse_args()

    hugg_data_val = Dataset.from_json(args.validation_files)
    pipe_dataset = MyDataset(hugg_data_val)
    print("Dataset loaded, length: ", len(pipe_dataset))

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
    ).eval()

    # merge new model
    if args.use_finetune_model:
        new_model = PeftModel.from_pretrained(base_model, args.new_model_path)
        new_model = new_model.merge_and_unload().eval()
    else:
        new_model = base_model

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = "[PAD]" #tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_pipe = pipeline(
        task="text-generation", 
        model=new_model, 
        tokenizer=tokenizer, 
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True, 
        temperature=0.95, 
        max_new_tokens=args.max_new_tokens, 
        repetition_penalty=1.15, 
        return_full_text=False)  
    
    infer_for_each_rank(
        model_pipe, 
        pipe_dataset, 
        args.local_rank, 
        args.batch_size_each_gpu, 
        args.pred_len, 
        args.output_path)


    if args.local_rank == 0:
        all_pred_file_path = [os.path.join(args.output_path, 'preds_' + str(i) + '.txt') for i in range(args.infer_times)]
        # loop check if all preds are saved
        while True:
            if all([os.path.exists(path) for path in all_pred_file_path]):
                break
            else:
                print("Waiting for all preds saved...")
                time.sleep(10)
        # save labels
        get_labels(hugg_data_val, args.pred_len, args.output_path)
        #### Evaluation ####
        # load label
        labels_path = os.path.join(args.output_path, 'labels.txt')
        labels = np.loadtxt(labels_path)
        # load and merge preds
        preds = load_and_merge_preds(args.output_path, args.infer_times)
        # eval
        eval(labels, preds, args.output_path, args.infer_times)