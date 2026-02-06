"""
处理 yambda 数据集，使用滑动窗口生成训练数据
- 滑动窗口最大长度为20
- 不划分history和target，直接输出整个窗口作为inters
"""
import argparse
import os
from tqdm import tqdm


def process_inter_file(input_path, output_path, max_window_len=20, his_sep=" "):
    """
    处理 .inter 文件，使用滑动窗口生成多个训练样本
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        max_window_len: 滑动窗口最大长度
        his_sep: item之间的分隔符
    """
    print(f"Processing {input_path}...")
    print(f"Max window length: {max_window_len}")
    
    inter_data = []
    user_count = 0
    
    with open(input_path, 'r') as f:
        # 读取header
        header = f.readline().strip()
        print(f"Header: {header}")
        
        for line in tqdm(f, desc="Reading users"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            user_id = parts[0]
            item_list_str = parts[1]
            items = item_list_str.split(' ')
            
            user_count += 1
            
            # 滑动窗口处理：对于每个位置i，取items[:i+1]的最后max_window_len个
            for i in range(1, len(items)):
                # 取从开始到i+1位置的items，然后截取最后max_window_len个
                history = items[:i+1]
                if max_window_len > 0:
                    history = history[-max_window_len:]
                
                one_data = {
                    "user_id": user_id,
                    "inters": his_sep.join(history)
                }
                inter_data.append(one_data)
    
    print(f"Processed {user_count} users")
    print(f"Generated {len(inter_data)} training instances")
    
    # 写入输出文件，保持原有格式
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        for data in tqdm(inter_data, desc="Writing"):
            f.write(f"{data['user_id']}\t{data['inters']}\n")
    
    print("Done.")
    return inter_data


def main():
    parser = argparse.ArgumentParser(description='Process yambda dataset with sliding window')
    parser.add_argument('--input', type=str, 
                        default='/home/hongminjie/MiniOneRec/yambda/sequential-multievent-500m/sequential-multievent-500m.inter',
                        help='Input .inter file path')
    parser.add_argument('--output', type=str,
                        default='/home/hongminjie/MiniOneRec/yambda/sequential-multievent-500m/sequential-multievent-500m-sft.inter',
                        help='Output .inter file path')
    parser.add_argument('--max_window_len', type=int, default=20,
                        help='Maximum sliding window length')
    
    args = parser.parse_args()
    
    process_inter_file(args.input, args.output, args.max_window_len)


if __name__ == '__main__':
    main()
