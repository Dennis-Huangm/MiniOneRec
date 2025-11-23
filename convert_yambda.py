from datasets import load_dataset
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

CONFIG = "flat-multievent-50m"    # 或 “flat‑multievent‑5b” 调试/生产时切换
OUT_REVIEWS = "reviews_yambda_extended.json"
OUT_META = "meta_yambda.json"

def generate_title(item_id, track_length_seconds):
    # 生成标题：例如 "Track i_821534 - 3 min 30 sec"
    if track_length_seconds is not None:
        minutes = track_length_seconds // 60
        seconds = track_length_seconds % 60
        return f"Track {item_id} - {minutes} min {seconds} sec"
    else:
        # 如果没有 track_length_seconds，返回简单的标题
        return f"Track {item_id}"

def convert_event(e):
    uid = e["uid"]
    item_id = e["item_id"]
    ts = int(e["timestamp"])
    is_org = e.get("is_organic", 1)
    played_ratio = e.get("played_ratio_pct", None)
    length_sec = e.get("track_length_seconds", None)
    event_type = e.get("event_type", None)

    return {
        "reviewerID": f"u_{uid}",
        "asin": f"i_{item_id}",
        "overall": 5,
        "unixReviewTime": ts,
        "reviewTime": "01 1, 2017",
        "summary": "",
        "reviewText": "",

        # Yambda 特有字段
        "orig_item_id": item_id,
        "is_organic": is_org,
        "played_ratio_pct": played_ratio,
        "track_length_seconds": length_sec,
        "event_type": event_type
    }

def process_batch(batch):
    # batch keys correspond to columns
    # We iterate over the number of rows in the batch
    # Find the length of the batch (number of rows)
    first_key = next(iter(batch))
    n_rows = len(batch[first_key])
    
    res_lines = []
    res_meta = []
    
    # List of keys to reconstruct the dict
    keys = list(batch.keys())
    
    for i in range(n_rows):
        # Reconstruct the row dict 'e'
        e = {k: batch[k][i] for k in keys}
        rec = convert_event(e)

        res_lines.append(json.dumps(rec, ensure_ascii=False))
        track_length_seconds = rec.get("track_length_seconds", None)
        # Store as structured data instead of dynamic dict keys
        res_meta.append({"asin": rec["asin"], "title": generate_title(rec["asin"], track_length_seconds)})
    
    return {"json_line": res_lines, "meta": res_meta}

def main():
    # 加载所有数据集（train、validation、test）
    dataset = load_dataset("yandex/yambda", CONFIG, split="train")
    
    asin_dic = {}
    n_in = 0
    n_out = 0

    # Use all available cores for dataset mapping
    n_proc = max(1, cpu_count())

    with open(OUT_REVIEWS, "w", encoding="utf-8") as fout:
        ds = dataset
        # Use dataset.map for efficient parallel processing
        # batched=True significantly reduces overhead
        # num_proc enables multiprocessing
        # remove_columns saves memory by dropping original data in the result
        processed = ds.map(
            process_batch,
            batched=True,
            batch_size=1000,
            num_proc=n_proc,
            remove_columns=ds.column_names,
            desc=f"Formatting train"
        )
        
        # Iterate over the processed dataset and write to file
        # processed is a Dataset with columns 'json_line' and 'asin'
        # We use iter with batch_size to efficiently read chunks
        for batch in tqdm(processed.iter(batch_size=5000), total=(len(processed) + 4999) // 5000, desc=f"Writing train"):
            lines = batch["json_line"]
            meta = batch["meta"]
            
            # Batch write and update
            fout.write("\n".join(lines) + "\n")
            
            # Update asin dictionary from the structured meta list
            for m in meta:
                asin_dic[m["asin"]] = m["title"]
            
            count = len(lines)
            n_in += count
            n_out += count


    print(f"Total read: {n_in}, written: {n_out}, unique items: {len(asin_dic)}")

    # 生成meta文件
    with open(OUT_META, "w", encoding="utf-8") as fmeta:
        for asin, title in tqdm(asin_dic.items(), desc="Writing meta"):
            meta = {
                "asin": asin,
                "title": title,
                "description": "",
                "brand": "",
                "categories": []
            }
            fmeta.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Meta file written: {OUT_META}")

if __name__ == "__main__":
    main()
