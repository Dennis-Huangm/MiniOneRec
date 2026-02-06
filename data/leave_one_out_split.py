#!/usr/bin/env python3
"""
Leave-One-Out split for sequential dataset
将每个用户序列的最后一个item作为测试集，其余作为训练集
"""
import os
import argparse
from tqdm import tqdm


def leave_one_out_split(input_file, output_dir=None):
    """
    对sequential数据进行留一法划分
    
    Args:
        input_file: 原始.inter文件路径
        output_dir: 输出目录，默认为输入文件所在目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # 获取基础文件名
    base_name = os.path.basename(input_file)
    if base_name.endswith('.inter'):
        base_name = base_name[:-6]  # 去掉 .inter 后缀
    
    train_file = os.path.join(output_dir, f"{base_name}.train.inter")
    test_file = os.path.join(output_dir, f"{base_name}.test.inter")
    
    print(f"Reading from: {input_file}")
    print(f"Writing train to: {train_file}")
    print(f"Writing test to: {test_file}")
    
    # 统计行数用于进度条
    print("Counting lines...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 处理数据
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(train_file, 'w', encoding='utf-8') as f_train, \
         open(test_file, 'w', encoding='utf-8') as f_test:
        
        # 读取并写入表头
        header = f_in.readline()
        f_train.write(header)
        f_test.write(header)
        
        # 处理每一行
        train_count = 0
        test_count = 0
        skip_count = 0
        
        for line in tqdm(f_in, total=total_lines-1, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            
            # 分割user_id和item_id_list
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Invalid line format, skipping: {line[:100]}")
                skip_count += 1
                continue
            
            user_id = parts[0]
            item_list = parts[1].split()
            
            # 如果序列长度小于2，跳过（至少需要1个训练+1个测试）
            if len(item_list) < 2:
                print(f"Warning: User {user_id} has less than 2 items, skipping")
                skip_count += 1
                continue
            
            # 分割训练集和测试集
            train_items = item_list[:-1]  # 除最后一个外的所有items
            test_item = item_list[-1]      # 最后一个item
            
            # 写入训练集
            train_line = f"{user_id}\t{' '.join(train_items)}\n"
            f_train.write(train_line)
            train_count += 1
            
            # 写入测试集
            test_line = f"{user_id}\t{test_item}\n"
            f_test.write(test_line)
            test_count += 1
    
    print(f"\n✓ Split completed!")
    print(f"  - Train samples: {train_count}")
    print(f"  - Test samples: {test_count}")
    print(f"  - Skipped lines: {skip_count}")
    print(f"\nOriginal file remains unchanged: {input_file}")


def main():
    parser = argparse.ArgumentParser(description='Leave-One-Out split for sequential dataset')
    parser.add_argument('--input_file', type=str, default='/home/hongminjie/MiniOneRec/yambda/sequential-multievent-500m/sequential-multievent-500m.inter', help='Input .inter file path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input file)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    leave_one_out_split(args.input_file, args.output_dir)


if __name__ == '__main__':
    main()
