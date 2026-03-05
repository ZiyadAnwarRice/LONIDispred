python - <<'PY'
import os, pickle, sys

def load_pkl(path):
    path = os.path.expandvars(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None

def check_split(name, seq_path, lab_path, msa_path):
    seq = load_pkl(seq_path)
    lab = load_pkl(lab_path)
    msa = load_pkl(msa_path)

    n_seq = safe_len(seq)
    n_lab = safe_len(lab)
    n_msa = safe_len(msa)

    ok_seq_lab = (n_seq == n_lab)
    ok_seq_msa = (n_seq == n_msa)

    status = "PASS" if (ok_seq_lab and ok_seq_msa) else "FAIL"

    print(f"\n[{name}] {status}")
    print(f"  sequences: {os.path.expandvars(seq_path)}  len={n_seq} type={type(seq).__name__}")
    print(f"  labels:    {os.path.expandvars(lab_path)}  len={n_lab} type={type(lab).__name__}")
    print(f"  msa:       {os.path.expandvars(msa_path)}  len={n_msa} type={type(msa).__name__}")

    if not ok_seq_lab:
        print(f"  !! len mismatch: sequences({n_seq}) != labels({n_lab})")
    if not ok_seq_msa:
        print(f"  !! len mismatch: sequences({n_seq}) != msa({n_msa})")

    return status == "PASS"

cfg = {
    "train": dict(
        seq="/work/$USER/LONIDispred/data/train_sequences.pkl",
        lab="/work/$USER/LONIDispred/data/train_labels.pkl",
        msa="/work/$USER/LONIDispred/MSA_list/msa_feat_F23_train_LIST.pkl",
    ),
    "val": dict(
        seq="/work/$USER/LONIDispred/data/testPDB_sequences.pkl",
        lab="/work/$USER/LONIDispred/data/testPDB_labels.pkl",
        msa="/work/$USER/LONIDispred/MSA_list/msa_feat_F23_testPDB_LIST.pkl",
    ),
    "testNox": dict(
        seq="/work/$USER/LONIDispred/data/Seq_CAID3NOX.pkl",
        lab="/work/$USER/LONIDispred/data/target_CAID3NOX.pkl",
        msa="/work/$USER/LONIDispred/MSA_list/msa_feat_F23_ValNOX_LIST.pkl",
    ),
    "testPDB": dict(
        seq="/work/$USER/LONIDispred/data/Seq_CAID3PDB.pkl",
        lab="/work/$USER/LONIDispred/data/target_CAID3PDB.pkl",
        msa="/work/$USER/LONIDispred/MSA_list/msa_feat_F23_ValPDB_LIST.pkl",
    ),
}

all_ok = True
for split, paths in cfg.items():
    all_ok &= check_split(split, paths["seq"], paths["lab"], paths["msa"])

print("\n====================")
print("OVERALL:", "ALL PASS ✅" if all_ok else "SOME FAIL ❌")
print("====================\n")

sys.exit(0 if all_ok else 1)
PY