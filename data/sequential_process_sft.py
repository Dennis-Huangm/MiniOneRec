import argparse
import os
import json
import pandas as pd
import collections
import multiprocessing
from tqdm import tqdm
import numpy as np
import gc

def _count_items_shard(ds_shard):
    """
    辅助函数：统计分片中的 Item 频率
    """
    cnt = collections.defaultdict(int)
    
    # 仅读取 item_ids 列，减少 IO
    # 注意：Dataset.iter() 不支持 columns 参数，需先 select_columns
    target_ds = ds_shard
    if hasattr(ds_shard, "select_columns"):
        target_ds = ds_shard.select_columns(["item_ids"])
        
    for batch in target_ds.iter(batch_size=1000):
        for seq in batch['item_ids']:
            for iid in seq:
                cnt[iid] += 1
    return cnt

# 新增：引入 datasets 库
try:
    from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)



def check_path(path):
    os.makedirs(path, exist_ok=True)

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def write_remap_index(index_map, file_path):
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")

# ==========================================
# 数据加载适配 (核心修改部分)
# ==========================================

def load_hf_data(dataset_name, input_path, max_samples=None):
    """
    加载 Hugging Face Dataset，保持 Dataset 对象以支持内存映射。
    避免使用 to_pandas() 以防止大文件 OOM。
    """
    print(f"Loading Hugging Face dataset from {dataset_name}...")
    
    try:
        # 1. 加载数据（Memory-mapped，不会立即读取到内存）
        dataset = load_dataset('parquet', data_dir=input_path)
        
        # 2. 处理 DatasetDict
        if isinstance(dataset, DatasetDict):
            print("Detected DatasetDict, using 'train' split...")
            if 'train' in dataset:
                ds = dataset['train']
            else:
                first_key = list(dataset.keys())[0]
                print(f"No 'train' split found, using '{first_key}'...")
                ds = dataset[first_key]
        else:
            ds = dataset

        # 3. 列名对齐
        # 注意：Dataset 是不可变的，rename_column 会返回新对象
        current_cols = ds.column_names
        if 'item_id' in current_cols:
            print("Renaming item_id -> item_ids")
            ds = ds.rename_column('item_id', 'item_ids')
        
        if 'timestamp' in current_cols:
            print("Renaming timestamp -> timestamps")
            ds = ds.rename_column('timestamp', 'timestamps')

        if max_samples is not None:
            print(f"DEBUG: selecting first {max_samples} samples for testing...")
            ds = ds.select(range(min(len(ds), max_samples)))
            
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure the path is a valid Hugging Face dataset directory.")
        exit(1)

# ==========================================
# K-Core 过滤逻辑
# ==========================================


def k_core_filtering(ds, user_k=5, item_k=5):
    print(f"\nStarting K-core filtering (User K={user_k}, Item K={item_k})...")
    print(f"Initial state: {len(ds)} users")
    
    # Valid items filtering setup
    valid_items_set = None
    if os.path.exists('valid_items.json'):
        print("Loading valid_items.json for filtering...")
        try:
            with open('valid_items.json', 'r') as f:
                valid_items_set = set(json.load(f))
            print(f"Loaded {len(valid_items_set)} valid items.")
        except Exception as e:
            print(f"Error loading valid_items.json: {e}")

    
    # 并行进程数
    num_proc = 30
    
    iteration = 0
    while True:
        iteration += 1
        prev_users_count = len(ds)
        
        # Step 1: 统计 Item 频率
        # 优化：根据数据量决定是否使用多进程统计
        # if len(ds) < 5000:
        #     item_counts = collections.defaultdict(int)
        #     for batch in tqdm(ds.iter(batch_size=1000), desc=f"Counting items iter {iteration}", leave=False):
        #         for seq in batch['item_ids']:
        #             for iid in seq:
        #                 item_counts[iid] += 1
        # else:
        print(f"  Iter {iteration}: Counting items (parallel, {num_proc} procs)...")
        shards = [ds.shard(num_shards=num_proc, index=i, contiguous=True) for i in range(num_proc)]
        
        with multiprocessing.Pool(num_proc) as pool:
            results = pool.map(_count_items_shard, shards)
        
        item_counts = collections.defaultdict(int)
        for res in results:
            for k, v in res.items():
                item_counts[k] += v
                
        if valid_items_set is not None and iteration == 1:
            keep_items = {iid for iid, count in item_counts.items() if count >= item_k and iid in valid_items_set}
        else:
            keep_items = {iid for iid, count in item_counts.items() if count >= item_k}
        num_items_dropped = len(item_counts) - len(keep_items)
        
        # Step 2: 过滤 Items
        if num_items_dropped > 0:
            print(f"  Iter {iteration}: Dropping {num_items_dropped} items (<{item_k})")
            
            def filter_items_batched(batch):
                new_ids = []
                new_times = []
                
                # 检查是否存在 track_length_seconds 列
                has_track_len = 'track_length_seconds' in batch
                
                for i in range(len(batch['item_ids'])):
                    items = batch['item_ids'][i]
                    times = batch['timestamps'][i]
                    
                    # 安全获取时长信息
                    track_len_seq = batch['track_length_seconds'][i] if has_track_len else None
                    
                    f_items, f_times = [], []
                    
                    for j, iid in enumerate(items):
                        if iid in keep_items:
                            f_items.append(iid)
                            f_times.append(times[j])
                            
                    new_ids.append(f_items)
                    new_times.append(f_times)
                    
                return {'item_ids': new_ids, 'timestamps': new_times}

            ds = ds.map(filter_items_batched, batched=True, batch_size=1000, desc=f"Filtering items iter {iteration}", num_proc=num_proc)
            
        # Step 3: 过滤 Users
        # 优化：启用多进程 filter
        ds_filtered = ds.filter(lambda x: len(x['item_ids']) >= user_k, desc=f"Filtering users iter {iteration}", num_proc=num_proc)
        num_users_dropped = prev_users_count - len(ds_filtered)
        print(f"  Iter {iteration}: Dropping {num_users_dropped} users (<{user_k})")
        ds = ds_filtered
        
        if num_items_dropped == 0 and num_users_dropped == 0:
            if len(keep_items) > 2000000:
                user_k += 1
                item_k += 1
                print(f"  Data count {keep_items} > 2M. Increasing user_k={user_k}, item_k={item_k}")
                continue
            print("K-core converged.")
            break
        if len(ds) == 0:
            break
            
    return ds



# ==========================================
# 并行处理辅助函数
# ==========================================

# 全局变量用于多进程共享 (避免 pickle 开销)
_shared_ds = None
_shared_u2i = None
_shared_i2i = None

def process_shard_sequences(args):
    """
    处理单个 shard 的序列生成
    """
    shard_idx, num_shards, batch_size = args
    
    # 子进程直接访问 copy-on-write 的全局变量
    # 注意：ds.shard 是 lazy 的，开销很小
    ds_shard = _shared_ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
    
    local_interactions = []
    
    # 显式选择列以减少 IO
    cols = ['uid', 'item_ids', 'timestamps']
        
    target_ds = ds_shard
    if hasattr(ds_shard, "select_columns"):
        target_ds = ds_shard.select_columns(cols)
    
    for batch in target_ds.iter(batch_size=batch_size):
        b_uids = batch['uid']
        b_item_seqs = batch['item_ids']
        b_time_seqs = batch['timestamps']
        
        # 2. 生成交互序列
        for i in range(len(b_uids)):
            original_uid = str(b_uids[i])
            if original_uid not in _shared_u2i:
                continue
                
            u_idx = _shared_u2i[original_uid]
            
            original_item_seq = b_item_seqs[i]
            original_time_seq = b_time_seqs[i]
            
            item_ids_remapped = []
            seq_times = []
            
            for k, iid in enumerate(original_item_seq):
                if iid in _shared_i2i:
                    item_ids_remapped.append(_shared_i2i[iid])
                    seq_times.append(original_time_seq[k])
            
            seq_len = len(item_ids_remapped)
            for k in range(0, seq_len, 20):
                chunk = item_ids_remapped[k : k + 20]
                
                # 只保留长度为20的chunk
                if len(chunk) != 20:
                    continue
                
                ts = seq_times[k + 19]  # 使用第20个item的时间戳
                
                # 优化：转换为 Tuple 和 预处理 String 以减少内存占用
                # history_seq 转为字符串存储，避免 list overhead
                seq_str = " ".join(map(str, chunk))
                
                local_interactions.append((u_idx, seq_str, ts))
            
    return local_interactions

# ==========================================
# Main
# ==========================================

def process_data(args):
    # 1. 加载 HF 数据 (修改点)
    ds = load_hf_data(args.dataset, args.input_path, args.debug_size)
    print(f"Loaded {len(ds)} users.")


    # 3. K-Core
    ds = k_core_filtering(ds, user_k=args.user_k, item_k=args.item_k)
    
    if len(ds) == 0:
        print("No data left.")
        return

    # 4. ID Map 构建
    print("Building ID maps...")
    uids = set()
    items = set()
    
    # 优化：仅读取需要的列
    target_ds = ds
    if hasattr(ds, "select_columns"):
        target_ds = ds.select_columns(['uid', 'item_ids'])

    # 使用迭代器快速收集 ID
    for batch in tqdm(target_ds.iter(batch_size=10000), total=(len(ds)//10000)+1, desc="Collecting IDs"):
        uids.update(str(u) for u in batch['uid'])
        for seq in batch['item_ids']:
            items.update(seq)
            
    user2index = {u: i for i, u in enumerate(sorted(list(uids)))}
    item2index = {i: idx for idx, i in enumerate(sorted(list(items)))} 
    
    print(f"Total Users: {len(user2index)}, Total Items: {len(item2index)}")

    # 5. 序列生成 (多进程并行优化)
    interaction_list = []
    
    # 设置全局变量，供子进程使用
    global _shared_ds, _shared_u2i, _shared_i2i
    _shared_ds = ds
    _shared_u2i = user2index
    _shared_i2i = item2index
    
    batch_size = 2000 
    num_workers = 10   # 降低并发数，避免峰值内存过高
    num_shards = 200   # 增加分片数，减少每个子进程持有的数据量
    
    print(f"Generating sequences (Parallel, {num_workers} workers, {num_shards} shards)...")
    
    # 只需要传递分片索引，不需要传递数据
    tasks = [(i, num_shards, batch_size) for i in range(num_shards)]
    
    with multiprocessing.Pool(num_workers) as pool:
        # 使用 imap_unordered 获取结果
        for local_inters in tqdm(pool.imap_unordered(process_shard_sequences, tasks), 
                                             total=num_shards, desc="Processing shards"):
            interaction_list.extend(local_inters)
            
    # 清理全局变量
    _shared_ds = None
    _shared_u2i = None
    _shared_i2i = None

    print("Sorting interactions by timestamp...")
    # Tuple 格式：(u_idx, seq_str, ts) -> 按 ts 排序 (index 2)
    interaction_list.sort(key=lambda x: x[2])
    
    # 6. 写入 Inter 文件
    print(f"Writing interaction files ({len(interaction_list)} samples)...")
    check_path(os.path.join(args.output_path, args.dataset))
    
    # 依然只生成一个 valid 文件
    file_path = os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter')
    
    with open(file_path, 'w') as f:
        f.write('user_id:token\titem_id_list:token_seq\n')
        for s in interaction_list:
            # s is (u_idx, seq_str, ts)
            f.write(f"{s[0]}\t{s[1]}\n")

    del interaction_list
    gc.collect()

    # 8. ID Maps
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))

    # 9. Embeddings Extraction
    emb_path = args.emb_path
    if os.path.exists(emb_path):
        print(f"Extracting embeddings from {emb_path}...")
        try:
            # Load embeddings dataset
            emb_ds = load_dataset('parquet', data_files=emb_path, split='train')
            
            # Determine embedding dimension
            sample_emb = emb_ds[0]['embed']
            emb_dim = len(sample_emb)
            print(f"Embedding dimension: {emb_dim}")
            
            # Initialize matrix with zeros
            num_items = len(item2index)
            emb_matrix = np.zeros((num_items, emb_dim), dtype=np.float16)
            print("Scanning parquet item_ids...")
            parquet_ids = emb_ds['item_id']
            
            # Create a set of valid original IDs for fast lookup
            valid_original_ids = set(item2index.keys())
            
            # Iterate over the embeddings dataset and fill the matrix
            # Note: This assumes item_id in parquet matcFhes keys in item2index
            count_found = 0
            
            relevant_indices = []
            
            # Iterate through all parquet IDs to find matches
            for i, iid in enumerate(tqdm(parquet_ids, desc="Indexing IDs")):
                if iid in valid_original_ids:
                    relevant_indices.append(i)
            
            print(f"Selected {len(relevant_indices)} rows from parquet.")
            
            # Fetch only relevant rows using the indices
            if relevant_indices:
                subset = emb_ds.select(relevant_indices)
                
                count_found = 0
                for row in tqdm(subset, desc="Extracting embeddings"):
                    iid = row['item_id']
                    if iid in item2index:
                        idx = item2index[iid]
                        emb = row['embed']
                        if len(emb) == emb_dim:
                             emb_matrix[idx] = np.array(emb, dtype=np.float16)
                             count_found += 1
            
            print(f"Found embeddings for {count_found}/{num_items} items.")
            
            # Save as .npy
            npy_path = os.path.join(args.output_path, args.dataset, f'{args.dataset}.item_emb.npy')
            np.save(npy_path, emb_matrix)
            print(f"Saved embedding matrix to {npy_path}")
            
        except Exception as e:
            print(f"Error processing embeddings: {e}")
    else:
        print(f"Embeddings file not found at {emb_path}")

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sequential-multievent-500m')
    # 这里输入应该是包含 DatasetDict 的文件夹路径
    parser.add_argument('--input_path', type=str, default='/home/hongminjie/datasets/yambda/sequential/500m/sharded/', help='Path to HF Dataset folder')
    parser.add_argument('--output_path', type=str, default='./yambda')
    parser.add_argument('--user_k', type=int, default=5)
    parser.add_argument('--item_k', type=int, default=5)
    parser.add_argument('--debug_size', type=int, default=None, help='Use only N samples for testing')
    parser.add_argument('--emb_path', type=str, default='/home/hongminjie/datasets/yambda/embeddings.parquet')
    args = parser.parse_args()
    process_data(args)