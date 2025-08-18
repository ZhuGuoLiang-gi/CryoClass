#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel, AutoTokenizer, pipeline

def download_prot_t5_xl_minimal(cache_dir: str):
    """
    下载 prot_t5_xl_uniref50 所需的最小文件到 cache_dir（使用国内镜像）
    文件包括：
        pytorch_model.bin
        config.json
        README.md
        special_tokens_map.json
        spiece.model
        tokenizer_config.json
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_id = "Rostlab/prot_t5_xl_uniref50"

    files_to_download = [
        "pytorch_model.bin",
        "config.json",
        "README.md",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json"
    ]

    print(f"正在使用镜像 {os.environ['HF_ENDPOINT']} 下载 {model_id} 的必要文件到 {cache_dir}")
    
    for filename in files_to_download:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=True,
            endpoint="https://hf-mirror.com"
        )
        print(f"下载完成: {path}")

    return cache_dir



def link_downloaded_files_to_model_dir(cache_dir: str, model_dir: str):
    """
    将 HuggingFace 下载的 prot_t5_xl_uniref50 文件软链接到指定的 model_dir
    """
    # 转为绝对路径
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    model_dir = os.path.abspath(os.path.expanduser(model_dir))

    os.makedirs(model_dir, exist_ok=True)

    # snapshots 目录，包含各个 commit 的模型快照
    snapshots_dir = os.path.join(cache_dir, "models--Rostlab--prot_t5_xl_uniref50", "snapshots")
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"找不到 snapshots 目录: {snapshots_dir}")

    # 取最新 snapshot
    snapshots = sorted(os.listdir(snapshots_dir))
    if not snapshots:
        raise FileNotFoundError(f"{snapshots_dir} 中没有 snapshot")
    
    latest_snapshot = os.path.join(snapshots_dir, snapshots[-1])
    if not os.path.exists(latest_snapshot):
        raise FileNotFoundError(f"最新 snapshot 路径不存在: {latest_snapshot}")

    # 需要软链接的文件
    files_to_link = [
        "pytorch_model.bin",
        "config.json",
        "README.md",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json"
    ]

    # 创建软链接
    for f in files_to_link:
        src = os.path.join(latest_snapshot, f)
        dst = os.path.join(model_dir, f)
        if not os.path.exists(src):
            raise FileNotFoundError(f"源文件不存在: {src}")
        if os.path.exists(dst):
            os.remove(dst)  # 覆盖旧软链接或文件
        os.symlink(src, dst)
        print(f"已创建软链接: {dst} -> {src}")

    return model_dir



def load_pipeline(model_dir: str):
    """
    从本地目录加载 prot_t5_xl_uniref50 并创建 feature-extraction pipeline
    """
    model = T5EncoderModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)  # <-- use slow tokenizer
    p = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    return p

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 prot_t5_xl_uniref50 所需文件并创建 pipeline")
    parser.add_argument("--cache_dir", default="../models/prot_t5_xl_uniref50", help="模型缓存目录")
    args = parser.parse_args()

    model_dir = download_prot_t5_xl_minimal(args.cache_dir)
    model_dir = link_downloaded_files_to_model_dir(args.cache_dir, model_dir)
    pipe = load_pipeline(model_dir)

    # 测试 pipeline
    test_sequence = ["MENSDSNDV"]  # 替换为实际蛋白序列
    features = pipe(test_sequence)
    print("提取特征完成，特征维度示例：", len(features[0][0]))
