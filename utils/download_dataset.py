import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time

ZENODO_API = "https://zenodo.org/api/records"

MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

def download_single(f, outdir):
    """下载单个文件，带进度条和重试"""
    fname = f["key"]
    url = f["links"]["self"]
    dest = os.path.join(outdir, fname)

    if os.path.exists(dest):
        return fname  # 已存在，跳过

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(dest, "wb") as f_out, tqdm(
                    total=total, unit="B", unit_scale=True, desc=fname
                ) as pbar:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f_out.write(chunk)
                            pbar.update(len(chunk))
            return fname
        except Exception as e:
            print(f"Attempt {attempt} failed for {fname}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise

def merge_parts(base, outdir="."):
    """合并 .partNNN 文件"""
    parts = sorted([f for f in os.listdir(outdir) if f.startswith(base + ".part")])
    if not parts:
        return
    merged_path = os.path.join(outdir, base)
    with open(merged_path, "wb") as outfile:
        for part in parts:
            part_path = os.path.join(outdir, part)
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
            os.remove(part_path)
    print(f"✅ Merged file: {merged_path}")

def download_all(files, outdir, max_workers=8):
    os.makedirs(outdir, exist_ok=True)
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for f in files:
            futures.append(executor.submit(download_single, f, outdir))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading file: {e}")

def main(deposition_id, outdir="downloads", max_workers=8):
    os.makedirs(outdir, exist_ok=True)
    
    # 获取 deposition 文件列表
    r = requests.get(f"{ZENODO_API}/{deposition_id}")
    r.raise_for_status()
    record = r.json()
    files = record["files"]

    # 多线程下载所有文件
    download_all(files, outdir, max_workers=max_workers)

    # 自动合并大文件分块
    part_bases = set()
    for f in os.listdir(outdir):
        if ".part" in f:
            base, _ = f.rsplit(".part", 1)
            part_bases.add(base)
    for base in part_bases:
        merge_parts(base, outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Zenodo deposition with multi-threading")
    # parser.add_argument("deposition_id", type=int, help="Zenodo deposition ID")
    parser.add_argument("--outdir", default="./dataset", help="下载目录")
    parser.add_argument("--workers", type=int, default=8, help="并行下载线程数")
    args = parser.parse_args()
    deposition_id=16899355
    main(deposition_id, args.outdir, args.workers)
