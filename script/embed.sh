python /root/T5.py \
  --fasta /root/autodl-tmp/Train/train_sequences.fasta \
  --outdir /root/autodl-tmp/embed \
  --model /root/autodl-tmp/T5/prot_t5 \
  --fp16 \
  --batch_size 32 \
  --window 1024 \
  --stride 512 \
  --chunk_batch_size 2 \
  --save_every 5000

python /root/T5.py \
  --fasta /root/autodl-tmp/cafa6/Test/testsuperset.fasta \
  --outdir /root/autodl-tmp/embed/T5_embed/test \
  --model /root/autodl-tmp/cafa6/prot_t5 \
  --fp16 \
  --batch_size 32 \
  --window 1024 \
  --stride 512 \
  --chunk_batch_size 2 \
  --save_every 5000

export HF_ENDPOINT="https://hf-mirror.com"
hf download facebook/esm2_t33_650M_UR50D --local-dir /root/autodl-tmp/cafa6/esm2 --exclude "*.h5" --exclude "tf_model*"

python ESM2.py \
  --fasta /root/autodl-tmp/cafa6/Train/train_sequences.fasta \
  --outdir /root/autodl-tmp/embed/esm_embed/train \
  --model /root/autodl-tmp/cafa6/esm2 \
  --fp16 \
  --local_only

python ESM2.py \
  --fasta /root/autodl-tmp/cafa6/Test/testsuperset.fasta \
  --outdir /root/autodl-tmp/embed/esm_embed/test \
  --model /root/autodl-tmp/cafa6/esm2 \
  --fp16 \
  --local_only