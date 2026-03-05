# convert_msa_dict_to_list.py
# Converts dict[id] -> [L,F] into list-aligned features using ids list order,
# and writes missing IDs to a text file.

import argparse
import pickle
import numpy as np

def load_ids(path: str):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_pkl", required=True, help="TXT file: one ID per line (sequence order)")
    ap.add_argument("--msa_dict_pkl", required=True, help="PKL: dict[id] -> np.ndarray [L,F]")
    ap.add_argument("--out_list_pkl", required=True, help="Output PKL: list aligned to ids order")
    ap.add_argument("--missing_out", default="", help="Optional output TXT for missing IDs")
    ap.add_argument("--strict", action="store_true", help="Fail if any ID missing in dict")
    args = ap.parse_args()

    ids = load_ids(args.ids_pkl)

    with open(args.msa_dict_pkl, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise ValueError("msa_dict_pkl must contain a dict[id] -> feature array")

    out = []
    missing_ids = []
    for pid in ids:
        feat = d.get(pid, None)
        if feat is None:
            missing_ids.append(pid)
            if args.strict:
                raise KeyError(f"Missing MSA feature for id={pid}")
            out.append(None)
        else:
            out.append(np.asarray(feat, dtype=np.float32))

    with open(args.out_list_pkl, "wb") as f:
        pickle.dump(out, f)

    # write missing ids
    if args.missing_out:
        with open(args.missing_out, "w") as f:
            for pid in missing_ids:
                f.write(f"{pid}\n")

    print(f"Done. wrote={args.out_list_pkl}  n={len(out)}  missing={len(missing_ids)}")
    if missing_ids:
        print("First missing IDs:", missing_ids[:20])
        if args.missing_out:
            print(f"Missing IDs written to: {args.missing_out}")

if __name__ == "__main__":
    main()