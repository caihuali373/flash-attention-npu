#!/usr/bin/env python3
import random
import pandas as pd
import sys

def gen_seqlen_list(max_seqlen: int, batch: int):
    return [random.randint(1, max_seqlen) for _ in range(batch)]

# 生成随机参数用例
def generate_cases(layout, num_cases=100):
    cases = []
    # 可能的headdim值
    qk_head_dims = [64, 128, 192, 256]
    
    for case_idx in range(num_cases):
        # 生成batch (batch < 20)
        batch = random.randint(1, 19)
        
        if layout == "BSND":
            seq_q = random.randint(1, 10240)
            seq_k = random.randint(1, 10240)
        elif layout == "TND":
            max_q_seqlen = random.randint(1, 8192)
            max_kv_seqlen = random.randint(1, 8192)
            list_seq_q = gen_seqlen_list(max_q_seqlen, batch)
            list_seq_kv = gen_seqlen_list(max_kv_seqlen, batch)

        # 生成headdim和headdim (两者相等)
        headdim = random.choice(qk_head_dims)
        
        # 生成nheads_k和nheads (nheads是nheads_k的整数倍)
        nheads_k = random.randint(1, 12)
        # nheads是nheads_k的整数倍，且不超过64
        multiple = random.randint(1, 4)  # 限制倍数，避免值过大
        nheads = nheads_k * multiple
        
        # window_size_left和window_size_right始终为-1
        window_size_left = -1
        window_size_right = -1
        
        # device和dtype固定
        dtype = random.choice(["half", "bf16"])
        is_deterministic = random.choice([0, 1])
        # keep_prob = random.random()
        keep_prob = 1
        is_causal = random.choice([0, 1])

        if layout == "BSND":
            case = {
                "Enable": "enable",
                "case_name": f"{layout}_case_{case_idx + 1:03d}",
                "batch": batch,
                "nheads": nheads,
                "nheads_k": nheads_k,
                "seq_q": seq_q,
                "seq_k": seq_k,
                "headdim": headdim,
                "window_size_left": window_size_left,
                "window_size_right": window_size_right,
                "layout": layout,
                "dtype": dtype,
                "is_deterministic": is_deterministic,
                "keep_prob": keep_prob,
                "is_causal": is_causal,
            }
        elif layout == "TND":
            case = {
                "Enable": "enable",
                "case_name": f"{layout}_case_{case_idx + 1:03d}",
                "batch": batch,
                "nheads": nheads,
                "nheads_k": nheads_k,
                "list_seq_q": list_seq_q,
                "list_seq_kv": list_seq_kv,
                "headdim": headdim,
                "window_size_left": window_size_left,
                "window_size_right": window_size_right,
                "layout": layout,
                "dtype": dtype,
                "is_deterministic": is_deterministic,
                "keep_prob": keep_prob,
                "is_causal": is_causal,
            }
        cases.append(case)

    return cases

# 保存用例到xlsx文件
def save_cases_to_xlsx(cases, filename="fag_cases_BSND_drop.xlsx"):
    df = pd.DataFrame(cases)
    df.to_excel(filename, index=False)
    print(f"已生成{len(cases)}个用例并保存到{filename}")

if __name__ == "__main__":
    layout = sys.argv[1]

    cases = generate_cases(layout, 220)
    save_cases_to_xlsx(cases, f"fag_cases_{layout}_drop.xlsx")
