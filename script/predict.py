#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
import joblib
import numpy as np
import os, argparse, json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from tool import run_command
import csv
import json
import random
import tempfile

def sample_fasta(input_fasta, n, temp_dir=None):
    """
    从输入 fasta 文件中随机抽取 n 条序列，保存到临时文件夹，返回临时文件路径
    """
    # 创建临时文件夹
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    # 读取所有序列
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if n > len(records):
        raise ValueError(f"Requested {n} sequences, but fasta has only {len(records)} sequences")

    # 随机抽样
    sampled_records = random.sample(records, n)

    # 临时文件路径
    temp_fasta_path = os.path.join(temp_dir, os.path.basename(input_fasta))
    
    # 保存抽样序列
    SeqIO.write(sampled_records, temp_fasta_path, "fasta")

    return temp_fasta_path


class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

def loading_model(model_file):
    input_size = 1024
    hidden_size = input_size // 2
    output_size = 2
    model = TwoLayerNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()
    return model


def embedding_protein(protT5_model,fasta_file, output_pkl, bs=10):
    cmd = f'python ../utils/ProtT5_embedding.py -f {fasta_file} -o {output_pkl} -b {bs} --protT5_model {protT5_model}'
    run_command(cmd)




def predict_fasta_embeddings(model, embedding_file):
    """
    对 embedding 矩阵进行批量预测，返回每行的预测结果。
    embedding_file: joblib 保存的列表，每个元素是 [seq_len, embedding_dim]
    """
    # 加载 embedding 矩阵
    protein_embeddings = joblib.load(embedding_file)  # list of [seq_len, embedding_dim] 或 [embedding_dim]
    protein_embeddings = np.array(protein_embeddings)
    fasta_embeddings = protein_embeddings.mean(axis=0)
    fasta_embeddings = torch.tensor(fasta_embeddings, dtype=torch.float32).unsqueeze(0)


    with torch.no_grad():
        logits = model(fasta_embeddings)  # [num_proteins, num_classes]
        probs = F.softmax(logits, dim=1).numpy()
        pred_classes = np.argmax(probs, axis=1)

    # 构建结果
    results = [{"prediction": int(pred), "confidence": float(prob[pred])} 
               for pred, prob in zip(pred_classes, probs)]

    return results


def main():
    parser = argparse.ArgumentParser(description="Protein classification from FASTA")
    parser.add_argument("-f", "--fasta", required=True, help="Input protein group FASTA file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output csv file")
    parser.add_argument("-m", "--model", default="../model/best_model_50.pth", help="Trained model file")
    parser.add_argument("--protT5_model", default="../model/prot_t5_xl_uniref50", help="Path to ProtT5 model directory")
    parser.add_argument("--sample_n", default="all", help="Number of sequences to randomly sample from each fasta (default 'all')")
    parser.add_argument("--embedding_outdir", default="./embeddings", help="Directory to save embeddings (pkl)")
    args = parser.parse_args()

    model = loading_model(args.model)
    os.makedirs(args.embedding_outdir, exist_ok=True)

    # 判断输入是文件还是目录
    fasta_files = []
    if os.path.isfile(args.fasta):
        fasta_files = [args.fasta]
    elif os.path.isdir(args.fasta):
        fasta_files = [os.path.join(args.fasta, f) for f in os.listdir(args.fasta) if f.endswith((".fasta", ".fa", ".faa"))]
    else:
        raise ValueError(f"{args.fasta} is not a valid file or directory")

    # 打开 CSV 文件写入结果
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fasta_name", "prediction", "confidence"])

        for fasta_file in fasta_files:
            fasta_name = os.path.splitext(os.path.basename(fasta_file))[0]

            # 判断是否随机抽样
            if args.sample_n != "all":
                try:
                    n = int(args.sample_n)
                    temp_fasta = sample_fasta(fasta_file, n)
                    fasta_to_use = temp_fasta
                except Exception as e:
                    print(f"Error sampling fasta {fasta_file}: {e}")
                    continue
            else:
                fasta_to_use = fasta_file

            # embedding 输出路径放到 embedding_outdir 下
            embedding_file = os.path.join(args.embedding_outdir, fasta_name + ".pkl")
            if not os.path.exists(embedding_file) or args.sample_n != "all":
                print(f">>>> generating embedding for {fasta_to_use}")
                embedding_protein(args.protT5_model, fasta_to_use, embedding_file, bs=10)

            results = predict_fasta_embeddings(model, embedding_file)
            
            for res in results:
                writer.writerow([fasta_name, res["prediction"], res["confidence"]])
    

if __name__ == "__main__":
    main()
