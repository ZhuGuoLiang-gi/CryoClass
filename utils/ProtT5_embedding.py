import sys
import pandas as pd
import numpy as np
import torch
import os
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import pickle
import math
from Bio import SeqIO
import argparse
from tool import batch_task
import torch.nn as nn
import joblib
import time
import tempfile
import shutil


def single_batch_ProtT5_embedding(sequence_list, batch_id, out_fpath, model_dpath=None, **kwargs):
    if os.path.exists(out_fpath):
        return 1

    # 格式化序列
    sequences_Example = [" ".join(list(seq)) for seq in sequence_list]

    tokenizer = T5Tokenizer.from_pretrained(model_dpath, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_dpath).eval()

    # 替换不支持的氨基酸
    sequence_inputs = [re.sub(r"[UZOB]", "X", example) for example in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequence_inputs, add_special_tokens=True, padding=True, return_tensors='pt')
    input_ids, attention_mask = ids['input_ids'], ids['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.cpu().numpy()
        torch.cuda.empty_cache()

    features = []
    for seq_num in range(len(embeddings)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embeddings[seq_num][:seq_len - 1]  # 去掉 <eos>
        features_pool = np.mean(seq_emd, axis=0)
        features.append(features_pool)

    with open(out_fpath, 'wb') as f:
        joblib.dump(features, f)

    # 释放显存和内存
    model.to('cpu')
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    torch.cuda.empty_cache()
    del model, input_ids, attention_mask
    print(f'\033[1;31mbatch id:{batch_id} completed\033[0m', flush=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ProtT5 embedding extractor')
    parser.add_argument('-f', help='input fasta file of protein', required=True)
    parser.add_argument('-o', help='output file name', required=True)
    parser.add_argument('-b', help='batch size', default=300, required=False)
    parser.add_argument('--tmp_dir', help='temporary directory for child outputs', required=False, default=None)
    parser.add_argument('--protT5_model', help='path to ProtT5 model directory', required=True)
    args = parser.parse_args()

    fasta_fpath = args.f
    out_fpath = args.o
    batch_size = int(args.b)
    model_dpath = args.protT5_model

    # 读取序列
    records = SeqIO.parse(fasta_fpath, "fasta")
    sequences = [str(record.seq) for record in records]

    # 划分 batch
    batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]

    # 临时目录
    if args.tmp_dir:
        tmp_dir = os.path.abspath(args.tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="protT5_", dir=os.path.dirname(out_fpath))

    tasks = []
    child_file_list = []
    out_name, suffix = os.path.splitext(os.path.basename(out_fpath))

    task_control = {
        'max_try': 0,
        'max_task_num_per_gpu': 1,
        'interval_output_tasks_info': 12,
        'gpu_max_load': 70,
        'requery_memory': 15000,
        'error_loop': False
    }

    for batch_id, batch_sequence in enumerate(batches):
        child_out_file = os.path.join(tmp_dir, f'{out_name}_{batch_size}_{batch_id}{suffix}')
        child_file_list.append(child_out_file)
        if not os.path.exists(child_out_file):
            task = {
                'task_name': f'{child_out_file}',
                'func': single_batch_ProtT5_embedding,
                'args': (batch_sequence, batch_id, child_out_file),
                'kwargs': {
                    'model_dpath': model_dpath,
                    'requery_memory': task_control['requery_memory'],
                    'gpu_device': None,
                    'gpu_max_usage': task_control['gpu_max_load'],
                    'max_tasks_num_per_gpu': task_control['max_task_num_per_gpu'],
                }
            }
            tasks.append(task)

    # 执行任务
    batch_task(tasks, **task_control)

    # 合并结果
    stop_delete = False
    try:
        combined_data = []
        for file in child_file_list:
            data = joblib.load(file)
            combined_data.extend(data)

        joblib.dump(combined_data, out_fpath)
    except Exception as e:
        print(f'{e}')
        stop_delete = True

    # 清理临时文件
    if not stop_delete:
        shutil.rmtree(tmp_dir, ignore_errors=True)
