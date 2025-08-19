import os
import aiohttp
import asyncio
from tqdm import tqdm
import argparse
from asyncio import Semaphore
import time

ZENODO_API = "https://zenodo.org/api/records"
MAX_RETRIES = 3
MIN_CHUNK_SIZE = 1 * 1024 * 1024  # 1MB
MAX_CHUNK_WORKERS = 8
MAX_FILE_WORKERS = 4



def compute_chunk_workers(file_size):
    if file_size < 10 * 1024 * 1024:  # <10MB
        return 1, 1
    elif file_size < 100 * 1024 * 1024:  # <100MB
        return min(4, file_size // MIN_CHUNK_SIZE), 2
    else:
        return min(MAX_CHUNK_WORKERS, file_size // MIN_CHUNK_SIZE), MAX_CHUNK_WORKERS

async def download_chunk(session, url, start, end, dest, sem, file_pbar, global_pbar, global_lock):
    headers = {"Range": f"bytes={start}-{end}"}
    chunk_file = f"{dest}.part{start}-{end}"

    if os.path.exists(chunk_file) and os.path.getsize(chunk_file) >= (end - start + 1):
        async with global_lock:
            global_pbar.update(end - start + 1)
        file_pbar.update(end - start + 1)
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem, session.get(url, headers=headers) as r:
                r.raise_for_status()
                with open(chunk_file, "ab") as f:
                    async for chunk in r.content.iter_chunked(8192):
                        f.write(chunk)
                        file_pbar.update(len(chunk))
                        async with global_lock:
                            global_pbar.update(len(chunk))
            return
        except Exception as e:
            print(f"Attempt {attempt} failed for chunk {start}-{end}: {e}")
            await asyncio.sleep(2)
    raise RuntimeError(f"Failed to download chunk {start}-{end}")

async def download_file(session, file_info, outdir, global_pbar, global_lock):
    fname = file_info["key"]
    url = file_info["links"]["self"]
    dest = os.path.join(outdir, fname)

    if os.path.exists(dest):
        async with global_lock:
            global_pbar.update(int(file_info.get("size", 0)))
        return

    async with session.head(url) as r:
        size = int(r.headers.get("Content-Length", 0))

    num_chunks, chunk_workers = compute_chunk_workers(size)
    chunk_size = max(MIN_CHUNK_SIZE, size // num_chunks)
    sem = Semaphore(chunk_workers)

    # 文件内部进度条
    with tqdm(total=size, unit="B", unit_scale=True, desc=fname, leave=True) as file_pbar:
        tasks = [
            download_chunk(session, url, start, min(start + chunk_size - 1, size - 1), dest, sem, file_pbar, global_pbar, global_lock)
            for start in range(0, size, chunk_size)
        ]
        await asyncio.gather(*tasks)

    # 合并分块
    with open(dest, "wb") as outfile:
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size - 1, size - 1)
            chunk_file = f"{dest}.part{start}-{end}"
            with open(chunk_file, "rb") as infile:
                outfile.write(infile.read())
            os.remove(chunk_file)

async def download_all_files(deposition_id, outdir):
    os.makedirs(outdir, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ZENODO_API}/{deposition_id}") as r:
            record = await r.json()
        files = record["files"]
        total_size = sum(int(f.get("size", 0)) for f in files)

        sem_files = Semaphore(MAX_FILE_WORKERS)
        global_lock = asyncio.Lock()
        start_time = time.time()

        # 全局进度条
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Total Progress") as global_pbar:

            async def sem_task(file_info):
                async with sem_files:
                    await download_file(session, file_info, outdir, global_pbar, global_lock)
                    # 更新全局速度与剩余时间
                    elapsed = time.time() - start_time
                    speed = global_pbar.n / elapsed / 1024 / 1024  # MB/s
                    remaining = (global_pbar.total - global_pbar.n) / (speed * 1024 * 1024) if speed > 0 else 0
                    global_pbar.set_postfix({
                        "Speed": f"{speed:.2f} MB/s",
                        "ETA": f"{int(remaining)}s"
                    })

            await asyncio.gather(*(sem_task(f) for f in files))

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_outdir = os.path.join(script_dir, "../dataset")


    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=default_outdir, help="下载目录")
    parser.add_argument("--deposition_id", type=int, default=16899355)
    args = parser.parse_args()

    asyncio.run(download_all_files(args.deposition_id, args.outdir))
