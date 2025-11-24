import argparse
import os
import json
import html
import re
import pandas as pd
import collections
from tqdm import tqdm

# ==========================================
# 文本清洗工具
# ==========================================

def clean_text(text):
    """Clean text by removing HTML tags and excessive whitespace"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', str(text))
    text = html.unescape(text)
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def check_path(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def write_json_file(data, file_path):
    """Write data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def write_remap_index(index_map, file_path):
    """Write index mapping to file"""
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")

# ==========================================
# K-Core 过滤逻辑 (新增核心部分)
# ==========================================

def filter_items_by_metadata(df, valid_asins):
    """
    根据 Metadata 过滤 Item (如果 Metadata 存在)
    仅保留在 valid_asins 中的 Item
    """
    if not valid_asins:
        return df
    
    print(f"Filtering items based on metadata ({len(valid_asins)} valid ASINs)...")
    
    new_item_ids = []
    new_timestamps = []
    
    # 将 valid_asins 转为 set 以加速查找 (注意类型统一转字符串)
    valid_set = set(str(x) for x in valid_asins)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Metadata filtering"):
        items = row['item_ids']
        times = row['timestamps']
        
        filtered_items = []
        filtered_times = []
        
        for i, item in enumerate(items):
            if str(item) in valid_set:
                filtered_items.append(item)
                filtered_times.append(times[i])
        
        new_item_ids.append(filtered_items)
        new_timestamps.append(filtered_times)
        
    df['item_ids'] = new_item_ids
    df['timestamps'] = new_timestamps
    
    # 移除变为空的用户
    df = df[df['item_ids'].apply(len) > 0].copy()
    print(f"After metadata filtering: {len(df)} users remain.")
    return df

def k_core_filtering(df, user_k=5, item_k=5):
    """
    针对 Sequential 数据的迭代式 K-core 过滤
    df: 包含 'uid', 'item_ids', 'timestamps'
    """
    print(f"\nStarting K-core filtering (User K={user_k}, Item K={item_k})...")
    print(f"Initial state: {len(df)} users")
    
    iteration = 0
    while True:
        iteration += 1
        prev_users_count = len(df)
        
        # --- Step 1: 统计 Item 频率 ---
        item_counts = collections.defaultdict(int)
        # 使用 chain 展平列表进行统计，或者直接遍历
        for items in df['item_ids']:
            for iid in items:
                item_counts[iid] += 1
                
        # 找出需要保留的 Items
        # 注意：为了处理不同类型 (int/str)，这里保持原始类型
        keep_items = {iid for iid, count in item_counts.items() if count >= item_k}
        num_items_dropped = len(item_counts) - len(keep_items)
        
        # 如果没有 Item 需要移除，且已经是第一次迭代之后（或者还没移除过用户），
        # 但为了保险起见，通常检查是否收敛的条件是：没有 Item 移除 AND 没有 User 移除
        
        # --- Step 2: 过滤 DataFrame 中的 Items 和 Timestamps ---
        if num_items_dropped > 0:
            print(f"  Iter {iteration}: Dropping {num_items_dropped} items (freq < {item_k})")
            
            new_item_ids = []
            new_timestamps = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Filtering items iter {iteration}"):
                items = row['item_ids']
                times = row['timestamps']
                
                f_items = []
                f_times = []
                
                for i, iid in enumerate(items):
                    if iid in keep_items:
                        f_items.append(iid)
                        f_times.append(times[i])
                
                new_item_ids.append(f_items)
                new_timestamps.append(f_times)
            
            df['item_ids'] = new_item_ids
            df['timestamps'] = new_timestamps
        else:
            print(f"  Iter {iteration}: No items dropped.")

        # --- Step 3: 过滤 Users (基于序列长度) ---
        # 计算新的序列长度
        df['seq_len'] = df['item_ids'].apply(len)
        
        # 移除长度小于 user_k 的用户
        df_filtered = df[df['seq_len'] >= user_k].copy()
        num_users_dropped = len(df) - len(df_filtered)
        
        print(f"  Iter {iteration}: Dropping {num_users_dropped} users (len < {user_k})")
        
        df = df_filtered
        current_users_count = len(df)
        
        # --- 收敛检查 ---
        if num_items_dropped == 0 and num_users_dropped == 0:
            print(f"K-core converged at iteration {iteration}.")
            break
            
        if current_users_count == 0:
            print("Warning: All users filtered out!")
            break
    
    print(f"Final state: {len(df)} users.")
    return df

# ==========================================
# Metadata 处理逻辑
# ==========================================

def load_metadata(metadata_file):
    """加载 Metadata 并提取有 Title 的 ASIN"""
    print(f"Loading metadata from {metadata_file}...")
    asin2meta = {}
    valid_asins = set()
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata"):
                try:
                    meta = json.loads(line)
                    # 逻辑参考 amazon18_data_process: 必须有 title 且长度 > 0
                    if 'title' in meta and len(str(meta['title']).strip()) > 0:
                        asin = meta['asin']
                        asin2meta[asin] = meta
                        valid_asins.add(asin)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Metadata file {metadata_file} not found.")
        return {}, set()
    
    print(f"Loaded {len(asin2meta)} valid metadata items.")
    return asin2meta, valid_asins

def create_item_features(valid_items_set, asin2meta, item2index):
    """生成 item.json"""
    item2feature = {}
    print("Generating item features...")
    for original_item_id in tqdm(valid_items_set, desc="Processing features"):
        original_id_str = str(original_item_id)
        if original_id_str in asin2meta:
            meta = asin2meta[original_id_str]
            new_id = item2index[original_item_id] # 注意这里用原始类型去查索引
            
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
# 主处理流程
# ==========================================

def process_data(args):
    # 1. 加载 Sequential 数据
    print(f"Loading sequential data from {args.sequential_file}...")
    if args.sequential_file.endswith('.parquet'):
        df = pd.read_parquet(args.sequential_file)
    elif args.sequential_file.endswith('.json'):
        df = pd.read_json(args.sequential_file, lines=True)
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .json")

    # 2. 加载 Metadata (如果提供)
    asin2meta = {}
    valid_metadata_asins = set()
    if args.metadata_file:
        asin2meta, valid_metadata_asins = load_metadata(args.metadata_file)
        # 2.1 Metadata 预过滤: 移除序列中没有 Metadata 的商品
        df = filter_items_by_metadata(df, valid_metadata_asins)

    # 3. K-Core 过滤 (新增)
    df = k_core_filtering(df, user_k=args.user_k, item_k=args.item_k)

    if len(df) == 0:
        print("Error: No data left after filtering.")
        return

    # 4. 构建映射和序列
    user2index = {}
    item2index = {}
    interaction_list = []
    valid_items_set = set() 
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating sequences"):
        original_uid = str(row['uid'])
        original_item_seq = row['item_ids'] 
        original_time_seq = row['timestamps']
        
        if original_uid not in user2index:
            user2index[original_uid] = len(user2index)
            
        item_ids_remapped = []
        for iid in original_item_seq:
            # 保持原始 key 用于 item2index 映射，如果是 int 就用 int，str 用 str
            # 这样可以兼容 parquet 里存的是 int 的情况
            if iid not in item2index:
                item2index[iid] = len(item2index) + 1 # ID 从 1 开始
            
            item_ids_remapped.append(item2index[iid])
            valid_items_set.add(iid)
            
        if len(item_ids_remapped) < 2:
            continue
            
        # 生成训练序列 (History -> Target)
        # 使用 8:1:1 切分逻辑，这里先收集所有交互，最后再 split
        for i in range(1, len(item_ids_remapped)):
            target_item = item_ids_remapped[i]
            st = max(i - 50, 0) # Max history 50
            history_seq = item_ids_remapped[st:i]
            
            interaction_list.append({
                'user_idx': user2index[original_uid],
                'history_seq': history_seq,
                'target_item': target_item,
                'timestamp': original_time_seq[i]
            })

    # 按时间排序 (对于时序推荐很重要)
    interaction_list.sort(key=lambda x: x['timestamp'])
    
    # 5. 写入数据文件 (8:1:1 分割)
    print(f"Writing interaction files ({len(interaction_list)} samples)...")
    check_path(os.path.join(args.output_path, args.dataset))
    
    total_len = len(interaction_list)
    split_1 = int(total_len * 0.8)
    split_2 = int(total_len * 0.9)
    
    datasets = {
        'train': interaction_list[:split_1],
        'valid': interaction_list[split_1:split_2],
        'test': interaction_list[split_2:]
    }
    
    for dtype, data in datasets.items():
        file_path = os.path.join(args.output_path, args.dataset, f'{args.dataset}.{dtype}.inter')
        with open(file_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for s in data:
                seq_str = " ".join(map(str, s['history_seq']))
                f.write(f"{s['user_idx']}\t{seq_str}\t{s['target_item']}\n")

    # 6. 生成 item.json (如果 Metadata 存在)
    if args.metadata_file and asin2meta:
        item_features = create_item_features(valid_items_set, asin2meta, item2index)
        item_json_path = os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json')
        write_json_file(item_features, item_json_path)

    # 7. 写入 ID 映射
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))
    
    print("Processing completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Amazon_Beauty')
    parser.add_argument('--sequential_file', type=str, required=True, help='Path to sequential data (.parquet or .json)')
    parser.add_argument('--metadata_file', type=str, default=None, help='Path to original metadata.json')
    parser.add_argument('--output_path', type=str, default='./data')
    # 新增 K-Core 参数
    parser.add_argument('--user_k', type=int, default=5, help='User k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='Item k-core filtering')
    
    args = parser.parse_args()
    
    process_data(args)