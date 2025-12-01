import polars as pl
import os
from tqdm import tqdm

# --- 配置 ---
input_file = '/home/huangminrui/datasets/yambda/sequential/500m/listens.parquet'
output_dir = '/home/huangminrui/datasets/yambda/sequential/500m/sharded/'
chunk_size = 25000  # 总共100万行，每次切2.5万行，会生成40个文件，每个约1GB

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def split_parquet_polars():
    print(f"正在使用 Polars 处理: {input_file}")
    
    # 1. 使用 scan_parquet 建立惰性计算图 (不会立即读取数据)
    # low_memory=True 会牺牲一点速度来换取更低的内存占用
    lf = pl.scan_parquet(input_file, low_memory=True)
    
    # 获取总行数 (利用之前 metadata 里的信息，或者快速计算)
    # 你日志里已经显示是 100000
    total_rows = 100000
    
    print(f"总行数: {total_rows}")
    print(f"目标切分: 每份 {chunk_size} 行")

    # 2. 循环切分并写入
    # Polars 的 slice 操作非常高效，它知道如何在 Parquet 块中寻址
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(num_chunks)):
        offset = i * chunk_size
        output_path = os.path.join(output_dir, f"part_{i:04d}.parquet")
        
        try:
            # slice(offset, length)
            # collect() 会触发实际读取，但只会把这一小块读入内存
            # 这里的 25000 行大约是 1GB 左右，内存应该能抗住
            df_chunk = lf.slice(offset, chunk_size).collect()
            
            # 写入新的 Parquet 文件
            # use_pyarrow=True 确保兼容性
            # compression='snappy' 是通用的压缩格式
            df_chunk.write_parquet(output_path, compression='snappy', use_pyarrow=True)
            
            # 释放内存（虽然 Python 会自动管理，显式删除是个好习惯）
            del df_chunk
            
        except Exception as e:
            print(f"\n❌ 处理第 {i} 块 (Offset {offset}) 时失败: {e}")
            # 如果内存爆了，可能需要进一步调小 chunk_size
            return

    print(f"\n✅ 切分完成！文件已保存在: {output_dir}")

if __name__ == "__main__":
    split_parquet_polars()  