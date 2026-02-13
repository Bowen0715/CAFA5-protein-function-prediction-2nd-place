# pip install obonet
# pip install pyvis
# pip install Bio
# pip install transformers
# pip install sentencepiece
import random
import numpy as np
from Bio import SeqIO
import torch
from transformers import T5Tokenizer, T5EncoderModel

# hf download Rostlab/prot_t5_xl_half_uniref50-enc --local-dir /root/autodl-tmp/T5/prot_t5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(
    "/root/autodl-tmp/T5/prot_t5",
    local_files_only=True,
    do_lower_case=False
)

model = T5EncoderModel.from_pretrained(
    "/root/autodl-tmp/T5/prot_t5",
    local_files_only=True,
).to(device)

if device.type == "cuda":
    model = model.half()

print("✅ ProtT5 loaded successfully!")

import re
def get_embeddings(seq: str) -> torch.Tensor:
    seq = re.sub(r"[UZOB]", "X", seq)
    seq_spaced = " ".join(list(seq))
    enc = tokenizer(
        seq_spaced,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False, 
        return_special_tokens_mask=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state[0]                  # (L_total, hidden)
        attn = enc["attention_mask"][0].bool()        # (L_total,)
        special = enc["special_tokens_mask"][0].bool()# (L_total,)
        keep = attn & (~special)
        h = h[keep]                                   # (L_res, hidden)
        emb = h.mean(dim=0)                            # (hidden,)
    return emb

a = get_embeddings('MRWQEMGYIFYPRKLR')

import tqdm
fn = '/root/autodl-tmp/Train/train_sequences.fasta'

sequences = SeqIO.parse(fn, "fasta")

ids = []
embeds = np.zeros((num_sequences, 1024))
i = 0
for seq in tqdm.tqdm(sequences):
    ids.append(seq.id)
    embeds[i] = get_embeddings(str(seq.seq)).detach().cpu().numpy()
    i += 1
    break #remove it for full calculation
        
np.save('train_embeds.npy', embeds)
np.save('train_ids.npy', np.array(ids))