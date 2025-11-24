import argparse
import os
import json
import html
import re
import pandas as pd
import collections
from tqdm import tqdm
# 新增：引入 datasets 库
try:
    from datasets import load_from_disk, Dataset, DatasetDict
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

# ==========================================
# 文本清洗工具
# ==========================================

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', str(text))
    text = html.unescape(text)
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

def load_hf_data(input_path):
    """
    加载 Hugging Face Dataset，保持 Dataset 对象以支持内存映射。
    避免使用 to_pandas() 以防止大文件 OOM。
    """
    print(f"Loading Hugging Face dataset from {input_path}...")
    
    try:
        # 1. 加载数据（Memory-mapped，不会立即读取到内存）
        dataset = load_from_disk(input_path)
        
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
            
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure the path is a valid Hugging Face dataset directory.")
        exit(1)

# ==========================================
# K-Core 过滤逻辑
# ==========================================

def filter_items_by_metadata(ds, valid_asins):
    if not valid_asins:
        return ds
    print(f"Filtering items based on metadata ({len(valid_asins)} valid ASINs)...")
    
    valid_set = set(str(x) for x in valid_asins)
    
    def filter_row_batch(batch):
        new_item_ids = []
        new_timestamps = []
        
        for i in range(len(batch['item_ids'])):
            items = batch['item_ids'][i]
            times = batch['timestamps'][i]
            
            f_items = []
            f_times = []
            for k, item in enumerate(items):
                if str(item) in valid_set:
                    f_items.append(item)
                    f_times.append(times[k])
            
            new_item_ids.append(f_items)
            new_timestamps.append(f_times)
            
        return {'item_ids': new_item_ids, 'timestamps': new_timestamps}

    # 使用 map 进行并行处理
    ds = ds.map(filter_row_batch, batched=True, desc="Metadata filtering", num_proc=4)
    
    # 过滤空序列
    ds = ds.filter(lambda x: len(x['item_ids']) > 0, desc="Dropping empty seqs")
    return ds

def k_core_filtering(ds, user_k=5, item_k=5):
    print(f"\nStarting K-core filtering (User K={user_k}, Item K={item_k})...")
    print(f"Initial state: {len(ds)} users")
    
    iteration = 0
    while True:
        iteration += 1
        prev_users_count = len(ds)
        
        # Step 1: 统计 Item 频率
        item_counts = collections.defaultdict(int)
        # 使用迭代器流式读取，避免一次性加载所有 item_ids 到内存
        for batch in tqdm(ds.iter(batch_size=10000), desc=f"Counting items iter {iteration}", leave=False):
            for seq in batch['item_ids']:
                for iid in seq:
                    item_counts[iid] += 1
                
        keep_items = {iid for iid, count in item_counts.items() if count >= item_k}
        num_items_dropped = len(item_counts) - len(keep_items)
        
        # Step 2: 过滤 Items
        if num_items_dropped > 0:
            print(f"  Iter {iteration}: Dropping {num_items_dropped} items (<{item_k})")
            
            def filter_items_batched(batch):
                new_ids = []
                new_times = []
                for i in range(len(batch['item_ids'])):
                    items = batch['item_ids'][i]
                    times = batch['timestamps'][i]
                    f_items, f_times = [], []
                    for j, iid in enumerate(items):
                        if iid in keep_items:
                            f_items.append(iid)
                            f_times.append(times[j])
                    new_ids.append(f_items)
                    new_times.append(f_times)
                return {'item_ids': new_ids, 'timestamps': new_times}

            ds = ds.map(filter_items_batched, batched=True, batch_size=1000, desc=f"Filtering items iter {iteration}", num_proc=30)
            
        # Step 3: 过滤 Users
        ds_filtered = ds.filter(lambda x: len(x['item_ids']) >= user_k, desc=f"Filtering users iter {iteration}")
        num_users_dropped = prev_users_count - len(ds_filtered)
        print(f"  Iter {iteration}: Dropping {num_users_dropped} users (<{user_k})")
        
        ds = ds_filtered
        
        if num_items_dropped == 0 and num_users_dropped == 0:
            print("K-core converged.")
            break
        if len(ds) == 0:
            break
            
    return ds

# ==========================================
# Metadata 读取
# ==========================================

def load_metadata(metadata_file):
    print(f"Loading metadata from {metadata_file}...")
    asin2meta = {}
    valid_asins = set()
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata"):
                try:
                    meta = json.loads(line)
                    if 'title' in meta and len(str(meta['title']).strip()) > 0:
                        asin = meta['asin']
                        asin2meta[asin] = meta
                        valid_asins.add(asin)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return {}, set()
    return asin2meta, valid_asins

def create_item_features(valid_items_set, asin2meta, item2index):
    item2feature = {}
    print("Generating item features...")
    for original_item_id in tqdm(valid_items_set, desc="Processing features"):
        original_id_str = str(original_item_id)
        if original_id_str in asin2meta:
            meta = asin2meta[original_id_str]
            new_id = item2index[original_item_id]
            
            title = clean_text(meta.get("title", ""))
            description = clean_text(meta.get("description", ""))
            brand = meta.get("brand", "").replace("by\n", "").strip()
            
            categories = meta.get("categories", [])
            cat_str = ""
            if categories:
                if isinstance(categories[0], list):
                    flat_categories = [c for group in categories for c in group]
                    cat_str = ",".join([str(c).strip() for c in flat_categories if "</span>" not in str(c)])
                else:
                    cat_str = ",".join([str(c).strip() for c in categories if "</span>" not in str(c)])
            
            item2feature[new_id] = {
                "title": title,
                "description": description,
                "brand": brand,
                "categories": cat_str
            }
    return item2feature

# ==========================================
# Main
# ==========================================

def process_data(args):
    # 1. 加载 HF 数据 (修改点)
    ds = load_hf_data(args.input_path)
    print(f"Loaded {len(ds)} users.")

    # 2. Metadata 过滤 (如果存在)
    asin2meta = {}
    valid_metadata_asins = set()
    if args.metadata_file:
        asin2meta, valid_metadata_asins = load_metadata(args.metadata_file)
        ds = filter_items_by_metadata(ds, valid_metadata_asins)

    # 3. K-Core
    ds = k_core_filtering(ds, user_k=args.user_k, item_k=args.item_k)
    
    if len(ds) == 0:
        print("No data left.")
        return

    # 4. 序列生成
    user2index, item2index = {}, {}
    interaction_list = []
    valid_items_set = set()
    
    # Dataset 直接迭代，不需要 iterrows
    for row in tqdm(ds, total=len(ds), desc="Generating sequences"):
        original_uid = str(row['uid'])
        original_item_seq = row['item_ids']
        original_time_seq = row['timestamps']
        
        if original_uid not in user2index:
            user2index[original_uid] = len(user2index)
            
        item_ids_remapped = []
        for iid in original_item_seq:
            if iid not in item2index:
                item2index[iid] = len(item2index) + 1
            item_ids_remapped.append(item2index[iid])
            valid_items_set.add(iid)
            
        if len(item_ids_remapped) < 2:
            continue
            
        for i in range(1, len(item_ids_remapped)):
            st = max(i - 50, 0)
            interaction_list.append({
                'user_idx': user2index[original_uid],
                'history_seq': item_ids_remapped[st:i],
                'target_item': item_ids_remapped[i],
                'timestamp': original_time_seq[i]
            })

    interaction_list.sort(key=lambda x: x['timestamp'])
    
    # 5. 写入 Inter 文件
    print(f"Writing interaction files ({len(interaction_list)} samples)...")
    check_path(os.path.join(args.output_path, args.dataset))
    
    total_len = len(interaction_list)
    split_1, split_2 = int(total_len * 0.8), int(total_len * 0.9)
    
    datasets = {
        # 'train': interaction_list[:split_1],
        # 'valid': interaction_list[split_1:split_2],
        'valid': interaction_list[:]
        # 'test': interaction_list[split_2:]
    }
    
    for dtype, data in datasets.items():
        file_path = os.path.join(args.output_path, args.dataset, f'{args.dataset}.{dtype}.inter')
        with open(file_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for s in data:
                seq_str = " ".join(map(str, s['history_seq']))
                f.write(f"{s['user_idx']}\t{seq_str}\t{s['target_item']}\n")

    # 6. Metadata Output
    if args.metadata_file and asin2meta:
        item_features = create_item_features(valid_items_set, asin2meta, item2index)
        write_json_file(item_features, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))

    # 7. ID Maps
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DatasetName')
    # 这里输入应该是包含 DatasetDict 的文件夹路径
    parser.add_argument('--input_path', type=str, required=True, help='Path to HF Dataset folder')
    parser.add_argument('--metadata_file', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./data')
    parser.add_argument('--user_k', type=int, default=5)
    parser.add_argument('--item_k', type=int, default=5)
    args = parser.parse_args()
    process_data(args)