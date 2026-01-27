# ---- read YAML with python ----
cd /root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place
BASE_PATH="/root/autodl-tmp/cafa6/"
RAPIDS_ENV_NAME="rapids-env/bin/python"
PYTORCH_ENV_NAME="pytorch-env/bin/python"

CONFIG_PATH="/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/config.yaml"
RAPIDS_ENV="${BASE_PATH}${RAPIDS_ENV_NAME}"
PYTORCH_ENV="${BASE_PATH}${PYTORCH_ENV_NAME}"

# ---- create envs ----
./create-rapids-env.sh "${BASE_PATH}"
./create-pytorch-env.sh "${BASE_PATH}"

# ---- parse fasta -> feather; create helpers ----
"${RAPIDS_ENV}" protlib/scripts/parse_fasta.py \
  --config-path "${CONFIG_PATH}"

awk 'BEGIN{OFS="\t"}
NR==1 {print; next}
{
  if ($3=="P") $3="BPO";
  else if ($3=="F") $3="MFO";
  else if ($3=="C") $3="CCO";
  print
}' /root/autodl-tmp/cafa6/Train/train_terms_ori.tsv \
> /root/autodl-tmp/cafa6/Train/train_terms.tsv

"${RAPIDS_ENV}" protlib/scripts/create_helpers.py \
  --config-path "${CONFIG_PATH}" \
  --batch-size 10000 -p

# TODO
if F; then
  # ---- download external go data ----
  "${RAPIDS_ENV}" protlib/scripts/downloads/dw_goant.py \
    --config-path "${CONFIG_PATH}"

  python /root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/protlib/scripts/downloads/dw_goant.py \
    --config-path /root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/config.yaml

  # ---- parse the files ----
  "${RAPIDS_ENV}" /root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/protlib/scripts/parse_go_single.py \
    --file /root/autodl-tmp/cafa6/temporal/goa_uniprot_all.gaf.216.gz \
    --config-path "${CONFIG_PATH}"

  "${RAPIDS_ENV}" /root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/protlib/scripts/parse_go_single.py \
    --file /root/autodl-tmp/cafa6/temporal/goa_uniprot_all.gaf.214.gz \
    --config-path "${CONFIG_PATH}" \
    --output old214

  # ---- propagate labels (train/test*) ----
  BASE_PATH="/root/autodl-tmp/cafa6/"
  TEMPORAL_DIR="${BASE_PATH}/temporal"
  LABEL_DIR="${TEMPORAL_DIR}/labels"
  GRAPH_OBO="${BASE_PATH}/Train/go-basic.obo"

  "${RAPIDS_ENV}" /root/test_env.py

  for f in "${LABEL_DIR}"/train* "${LABEL_DIR}"/test*; do
    bn="$(basename "$f")"
    out="${LABEL_DIR}/prop_${bn}"
    "${RAPIDS_ENV}" "/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/protlib/scripts/prop_tsv.py" \
      --path "${f}" \
      --graph "${GRAPH_OBO}" \
      --output "${out}" \
      --device 0 \
      --batch_size 30000 \
      --batch_inner 5000
  done

  # ---- create datasets + propagate quickgo51.tsv ----
  "${RAPIDS_ENV}" "/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/protlib/scripts/reproduce_mt.py" \
    --path "${TEMPORAL_DIR}" \
    --graph "${GRAPH_OBO}"

  "${RAPIDS_ENV}" "/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/protlib/scripts/prop_tsv.py" \
    --path "${TEMPORAL_DIR}/quickgo51.tsv" \
    --graph "${GRAPH_OBO}" \
    --output "${TEMPORAL_DIR}/prop_quickgo51.tsv" \
    --device 0 \
    --batch_size 30000 \
    --batch_inner 5000
fi

# ---- prepare NN solution ----
BASE_PATH="/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place"
"${PYTORCH_ENV}" "${BASE_PATH}/nn_solution/prepare.py" \
  --config-path "${CONFIG_PATH}"

mkdir -p embeds

export HF_ENDPOINT="https://hf-mirror.com"
hf download facebook/esm2_t33_650M_UR50D --local-dir /root/autodl-tmp/cafa6/esm2 --exclude "*.h5" --exclude "tf_model*"

# ---- embeddings (ProtT5 + ESM2 small) ----
BASE_PATH="/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place"
PYTORCH_ENV="/root/autodl-tmp/cafa6/pytorch-env/bin/python"
CONFIG_PATH="${BASE_PATH}/config.yaml"

"${PYTORCH_ENV}" "${BASE_PATH}/nn_solution/t5.py" \
  --config-path "${CONFIG_PATH}" 
"${PYTORCH_ENV}" "${BASE_PATH}/nn_solution/esm2sm.py" \
  --config-path "${CONFIG_PATH}" 

cd /root/autodl-tmp/cafa6
mkdir -p models

"${PYTORCH_ENV}" "${BASE_PATH}/memmap_to_npy.py" \
  --memmap /root/autodl-tmp/embed/T5_embed/test/embeddings.memmap.float16 \
  --out    /root/autodl-tmp/cafa6/embeds/t5/test_embeds.npy

"${RAPIDS_ENV}" "${BASE_PATH}/make_ids_from_feather.py" \
  --feather /root/autodl-tmp/cafa6/helpers/fasta/train_seq.feather \
  --id-col EntryID \
  --out /root/autodl-tmp/cafa6/embeds/esm_small/train_ids.npy

"${RAPIDS_ENV}" - <<'PY'
  import numpy as np
  import pandas as pd
  import os

  def save_unicode_ids(feather, out, id_col="EntryID"):
      df = pd.read_feather(feather)
      ids = df[id_col].astype(str).to_numpy(dtype="U")  # Unicode array, not object
      os.makedirs(os.path.dirname(out), exist_ok=True)
      np.save(out, ids)
      print("saved:", out, ids.shape, ids.dtype, "maxlen", max(map(len, ids)))

  for dataset in ["train", "test"]:
      feather = f"/root/autodl-tmp/cafa6/helpers/fasta/{dataset}_seq.feather"

      save_unicode_ids(feather, f"/root/autodl-tmp/cafa6/embeds/esm_small/{dataset}_ids.npy")
      save_unicode_ids(feather, f"/root/autodl-tmp/cafa6/embeds/t5/{dataset}_ids.npy")
PY
mamba activate /root/autodl-tmp/cafa6/pytorch-env
mamba activate /root/autodl-tmp/cafa6/rapids-env

# ---- train PB models ----
RAPIDS_ENV="/root/autodl-tmp/cafa6/rapids-env/bin/python"
for model_name in pb_t54500_raw pb_t54500_cond pb_t5esm4500_raw pb_t5esm4500_cond; do
  echo "[INFO] Training ${model_name}"
  "${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/train_pb.py" \
    --config-path "${CONFIG_PATH}" \
    --model-name "${model_name}" \
    --device 0 \
    > "/root/autodl-tmp/cafa6/logs/${model_name}.log" 2>&1
done

"${PYTORCH_ENV}" "${BASE_PATH}/make_sub_from_test_pred.py"

export MODEL_ROOT=/root/autodl-tmp/cafa6/models
for model_name in pb_t54500_raw pb_t54500_cond pb_t5esm4500_raw pb_t5esm4500_cond; do
  ls ${MODEL_ROOT}/${model_name}/model_*.pkl 2>/dev/null | \
    xargs -P 10 zstd -19 --rm
done

for model_name in pb_t54500_raw pb_t54500_cond pb_t5esm4500_raw pb_t5esm4500_cond; do
  find "${MODEL_ROOT}/${model_name}" -maxdepth 1 -name "model_*.pkl.zst" -print0 \
  | xargs -0 -n 1 -P 4 bash -c '
      f="$1"
      out="${f%.zst}.xz"
      echo "[INFO] $f -> $out"
      zstd -dc "$f" | xz > "$out" && rm -f "$f"
    ' _
done

# ---- train linear models ----
BASE_PATH="/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place"
RAPIDS_ENV="/root/autodl-tmp/cafa6/rapids-env/bin/python"
CONFIG_PATH="${BASE_PATH}/config.yaml"
# for model_name in lin_t5_raw lin_t5_cond; do
for model_name in lin_t5_cond; do
  echo "[INFO] Training ${model_name}"
  "${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/train_lin.py" \
    --config-path "${CONFIG_PATH}" \
    --model-name "${model_name}" \
    --device 0 \
    > "/root/autodl-tmp/cafa6/logs/${model_name}.log" 2>&1
done

# ---- train/infer stack models ----
"${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/create_gkf.py" \
  --config-path "${CONFIG_PATH}"

"${PYTORCH_ENV}" "${BASE_PATH}/nn_solution/train_models.py" \
  --config-path "${CONFIG_PATH}" \
  --device 0 \
  > "/root/autodl-tmp/cafa6/logs/nn_train.log" 2>&1

"${PYTORCH_ENV}" "${BASE_PATH}/nn_solution/inference_models.py" \
  --config-path "${CONFIG_PATH}" \
  --device 0 \
  > "/root/autodl-tmp/cafa6/logs/nn_train.log" 2>&1


"${PYTORCH_ENV}" "${BASE_PATH}/nn_solution/make_pkl.py" \
  --config-path "${CONFIG_PATH}"

# ---- train GCN for bp/mf/cc ----
BASE_PATH="/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place"
RAPIDS_ENV="/root/autodl-tmp/cafa6/rapids-env/bin/python"
CONFIG_PATH="${BASE_PATH}/config.yaml"

for ont in bp mf cc; do
  "${PYTORCH_ENV}" "${BASE_PATH}/protnn/scripts/train_gcn.py" \
    --config-path "${CONFIG_PATH}" \
    --ontology "${ont}" \
    --device 0
done

# ---- predict GCN ----
"${PYTORCH_ENV}" "${BASE_PATH}/protnn/scripts/predict_gcn.py" \
  --config-path "${CONFIG_PATH}" \
  --device "${DEVICE}"

# ---- postproc / submission ----
"${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/postproc/collect_ttas.py" \
  --config-path "${CONFIG_PATH}" \
  --device "${DEVICE}"

"${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/postproc/step.py" \
  --config-path "${CONFIG_PATH}" \
  --device "${DEVICE}" \
  --batch_size 30000 \
  --batch_inner 3000 \
  --lr 0.7 \
  --direction min

"${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/postproc/step.py" \
  --config-path "${CONFIG_PATH}" \
  --device "${DEVICE}" \
  --batch_size 30000 \
  --batch_inner 3000 \
  --lr 0.7 \
  --direction max

"${RAPIDS_ENV}" "${BASE_PATH}/protlib/scripts/postproc/make_submission.py" \
  --config-path "${CONFIG_PATH}" \
  --device "${DEVICE}" \
  --max-rate 0.5

# ---- preview ----
head "${BASE_PATH}/sub/submission.tsv"
