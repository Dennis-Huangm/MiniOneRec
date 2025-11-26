#!/usr/bin/env python3
"""
Convert dataset (Office/Industrial_and_Scientific) to MiniOneRec format with semantic IDs
"""

import csv
import sys
import json
import os
import argparse
import subprocess
from typing import Dict, List, Any
from tqdm import tqdm

def load_dataset(data_dir: str, dataset_name: str) -> Dict[str, Any]:
    """Load dataset metadata (items and semantic mapping), excluding splits"""
    data = {}
    
    # Load item metadata (id -> {title, description, ...})
    print(f"Loading item metadata...")
    with open(os.path.join(data_dir, f'{dataset_name}.item.json'), 'r') as f:
        data['items'] = json.load(f)
    print(f"Loaded {len(data['items'])} items")

    # Load item_id to semantic tokens mapping from index.json
    print(f"Loading semantic index...")
    with open(os.path.join(data_dir, f'{dataset_name}.faiss-rq.index.json'), 'r') as f:
        data['item_to_semantic'] = json.load(f)
    print(f"Loaded {len(data['item_to_semantic'])} item to semantic tokens mappings")
    
    # Note: We do NOT load splits here anymore to avoid OOM
    data['splits'] = {} 
    return data

def semantic_tokens_to_id(tokens: List[str]) -> str:
    """Convert semantic tokens list to concatenated string with brackets preserved"""
    # Keep brackets and concatenate directly (no spaces)
    return ''.join(tokens)

def create_item_info_file(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    """Create item info file (sid -> title -> item_id mapping)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item_id, item_data in tqdm(items.items(), desc="Creating item info file"):
            # Get semantic ID from index mapping
            if item_id in item_to_semantic:
                semantic_tokens = item_to_semantic[item_id]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                # Get item title, fallback to Item_id if not available
                item_title = item_data.get('title', f'Item_{item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{item_id}\n")

def convert_split_stream(split_name: str, input_file: str, items: Dict[str, Dict], 
                        item_to_semantic: Dict[str, List], output_dir: str, category: str = "Office_Products",
                        max_samples: int = None, seed: int = 42, keep_longest_only: bool = True):
    """
    Convert interaction data to MiniOneRec CSV format using streaming to avoid OOM.
    Reads input line-by-line and writes to output (or samples).
    """
    import random
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{category}_5_2016-10-2018-11.csv')
    
    print(f"Converting {split_name} data from {input_file}...")
    print(f"Output will be saved to {output_file}")

    # Field names for CSV
    fieldnames = ['user_id', 'history_item_title', 'item_title', 'history_item_id', 
                  'item_id', 'history_item_sid', 'item_sid']

    user_to_longest = {}  # For train data only
    reservoir = []        # For reservoir sampling
    reservoir_count = 0   # Total items seen for reservoir sampling
    
    # Helper to process a single line
    def process_line(line_parts):
        if len(line_parts) != 3:
            return None
            
        user_id, item_sequence, target_item = line_parts
        
        # Parse item sequence
        if item_sequence.strip():
            history_item_ids = [int(x) for x in item_sequence.split()]
        else:
            history_item_ids = []
        
        target_item_id = int(target_item)
        
        # Convert item_ids to semantic_ids
        history_semantic_ids = []
        for item_id in history_item_ids:
            if str(item_id) in item_to_semantic:
                history_semantic_ids.append(semantic_tokens_to_id(item_to_semantic[str(item_id)]))
        
        # Get target semantic_id
        target_semantic_id = None
        if str(target_item_id) in item_to_semantic:
            target_semantic_id = semantic_tokens_to_id(item_to_semantic[str(target_item_id)])
        
        if target_semantic_id is None:
            return None
        
        # Get item titles
        history_item_titles = []
        for item_id in history_item_ids:
            if str(item_id) in items:
                history_item_titles.append(items[str(item_id)].get('title', f'Item_{item_id}'))
        
        target_title = items.get(str(target_item_id), {}).get('title', f'Item_{target_item_id}')
        
        return {
            'user_id': f'A{user_id}',
            'history_item_title': str(history_item_titles),  # Convert list to string representation for CSV
            'item_title': target_title,
            'history_item_id': str(history_item_ids),
            'item_id': target_item_id,
            'history_item_sid': str(history_semantic_ids),
            'item_sid': target_semantic_id
        }

    # Open files
    with open(input_file, 'r') as f_in:
        # Setup CSV writer if we are writing directly (no sampling/aggregation needed yet)
        # If keep_longest_only is True (train) or max_samples is set, we delay writing.
        direct_write = not (keep_longest_only and split_name == 'train') and (max_samples is None)
        
        f_out = None
        writer = None
        
        if direct_write:
            f_out = open(output_file, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

        # Skip header
        header = f_in.readline()
        
        # Get total line count for progress bar
        try:
            result = subprocess.run(['wc', '-l', input_file], capture_output=True, text=True)
            total_lines = int(result.stdout.split()[0]) - 1  # Subtract 1 for header
        except:
            total_lines = None
        
        pbar = tqdm(desc=f"Processing {split_name}", unit="lines", total=total_lines)
        
        processed_count = 0
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            row = process_line(line.split('\t'))
            if not row:
                continue
                
            processed_count += 1
            pbar.update(1)

            # Logic: Train set keep longest
            if split_name == 'train' and keep_longest_only:
                uid = row['user_id']
                # Estimate sequence length from the history string or re-parse? 
                # process_line returns stringified list for CSV, let's parse it back or calculate before stringifying.
                # Optim: Calculate length from history_item_ids string (count commas + 1)
                # Or better: pass length out of process_line? 
                # Let's just count items in history_item_id string representation to be safe, or simple approximation
                # history_item_id is like "[1, 2, 3]"
                seq_len = row['history_item_id'].count(',') + 1 if len(row['history_item_id']) > 2 else 0
                
                if uid not in user_to_longest or seq_len > user_to_longest[uid][1]:
                    user_to_longest[uid] = (row, seq_len)
            
            # Logic: Sampling
            elif max_samples is not None:
                # Reservoir sampling
                if len(reservoir) < max_samples:
                    reservoir.append(row)
                else:
                    j = random.randint(0, reservoir_count)
                    if j < max_samples:
                        reservoir[j] = row
                reservoir_count += 1
            
            # Logic: Direct Write
            else:
                writer.writerow(row)

        pbar.close()

        # Write buffered data if any
        if not direct_write:
            # Open file now
            f_out = open(output_file, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            
            if split_name == 'train' and keep_longest_only:
                print(f"Writing {len(user_to_longest)} longest user sequences...")
                for row, _ in tqdm(user_to_longest.values(), desc="Writing"):
                    writer.writerow(row)
            
            elif max_samples is not None:
                print(f"Writing {len(reservoir)} sampled rows...")
                for row in tqdm(reservoir, desc="Writing"):
                    writer.writerow(row)
                    
        if f_out:
            f_out.close()

    print(f"Created {split_name} file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset (Office_Products/Industrial_and_Scientific) to MiniOneRec format with semantic IDs')
    parser.add_argument('--data_dir', type=str, default='yambda/sequential-multievent-500m',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='sequential-multievent-500m',
                       help='Dataset name (Office_Products, Industrial_and_Scientific)')
    parser.add_argument('--output_dir', type=str, default='yambda/sequential-multievent-500m',
                       help='Output directory for MiniOneRec format data')
    parser.add_argument('--category', type=str, default=None,
                       help='Category name for output files (if None, will use dataset_name)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                       help='Maximum number of samples to keep in validation set (None for all)')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='Maximum number of samples to keep in test set (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--keep_longest_only', action='store_true', default=False,
                       help='Keep only longest sequence per user in train data (default: False)')
    
    args = parser.parse_args()
    
    if args.category is None:
        args.category = args.dataset_name
    
    print(f"Loading {args.dataset_name} data from {args.data_dir}")
    dataset_data = load_dataset(args.data_dir, args.dataset_name)
    
    print(f"Found {len(dataset_data['items'])} items")
    
    # Create output directories
    for subdir in ['valid', 'info']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # Create item info file
    info_file = os.path.join(args.output_dir, 'info', f'{args.category}_5_2016-10-2018-11.txt')
    create_item_info_file(dataset_data['items'], dataset_data['item_to_semantic'], info_file)
    print(f"Created item info file: {info_file}")
    
    # Convert splits using streaming
    # You can add 'train' and 'test' to this list if needed
    splits_to_process = ['valid'] 
    
    for split_name in splits_to_process:
        split_file = os.path.join(args.data_dir, f'{args.dataset_name}.{split_name}.inter')
        if os.path.exists(split_file):
            max_samples = None
            if split_name == 'valid':
                max_samples = args.max_valid_samples
            elif split_name == 'test':
                max_samples = args.max_test_samples
                
            convert_split_stream(
                split_name,
                split_file,
                dataset_data['items'],
                dataset_data['item_to_semantic'],
                os.path.join(args.output_dir, split_name),
                args.category,
                max_samples=max_samples,
                seed=args.seed,
                keep_longest_only=args.keep_longest_only
            )
        else:
            print(f"Warning: Split file {split_file} not found")
    
    print(f"\nConversion completed! Data saved to {args.output_dir}")


if __name__ == '__main__':
    main()
