# Make an output folder for list-aligned MSA features
mkdir -p /ddnB/work/wasicse/LONIDispred/MSA_list

# Train
python /ddnB/work/wasicse/LONIDispred/convert_msa_dict_to_list.py \
  --ids_pkl /ddnB/work/wasicse/LONIDispred/data/id_train.txt \
  --msa_dict_pkl /ddnB/work/wasicse/LONIDispred/MSA/msa_feat_F23_by_id_trainWithMissing.pkl \
  --out_list_pkl /ddnB/work/wasicse/LONIDispred/MSA_list/msa_feat_F23_train_LIST.pkl \
  --strict

# Test NOX
python /ddnB/work/wasicse/LONIDispred/convert_msa_dict_to_list.py \
  --ids_pkl /ddnB/work/wasicse/LONIDispred/data/id_testNOX.txt \
  --msa_dict_pkl /ddnB/work/wasicse/LONIDispred/MSA/msa_feat_F23_by_id_testNOX.pkl \
  --out_list_pkl /ddnB/work/wasicse/LONIDispred/MSA_list/msa_feat_F23_testNOX_LIST.pkl \
  --strict

# Test PDB
python /ddnB/work/wasicse/LONIDispred/convert_msa_dict_to_list.py \
  --ids_pkl /ddnB/work/wasicse/LONIDispred/data/id_testPDB.txt \
  --msa_dict_pkl /ddnB/work/wasicse/LONIDispred/MSA/msa_feat_F23_by_id_testPDB.pkl \
  --out_list_pkl /ddnB/work/wasicse/LONIDispred/MSA_list/msa_feat_F23_testPDB_LIST.pkl \
  --strict

# Val NOX
python /ddnB/work/wasicse/LONIDispred/convert_msa_dict_to_list.py \
  --ids_pkl /ddnB/work/wasicse/LONIDispred/data/CAID3_NOX_ids.txt \
  --msa_dict_pkl /ddnB/work/wasicse/LONIDispred/MSA/msa_feat_F23_by_id_ValNOXShortList.pkl \
  --out_list_pkl /ddnB/work/wasicse/LONIDispred/MSA_list/msa_feat_F23_ValNOX_LIST.pkl \
  --strict

# Val PDB
python /ddnB/work/wasicse/LONIDispred/convert_msa_dict_to_list.py \
  --ids_pkl /ddnB/work/wasicse/LONIDispred/data/CAID3_PDB_ids.txt \
  --msa_dict_pkl /ddnB/work/wasicse/LONIDispred/MSA/msa_feat_F23_by_id_ValPDBShortList.pkl \
  --out_list_pkl /ddnB/work/wasicse/LONIDispred/MSA_list/msa_feat_F23_ValPDB_LIST.pkl \
  --strict