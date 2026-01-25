import argparse
import os
import re

import numpy as np
import torch
import tqdm
import yaml
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str, default="0")

def chunk_sequence(seq: str, window: int = 1024, stride: int = 512):
    seq = re.sub(r"[UZOB]", "X", seq)
    n = len(seq)
    if n <= window:
        return [seq]
    chunks = []
    for start in range(0, n, stride):
        chunk = seq[start:start + window]
        if len(chunk) < 50:  # 太短的尾巴可以丢掉（你也可以保留）
            break
        chunks.append(chunk)
        if start + window >= n:
            break
    return chunks

def get_embeddings(model, tokenizer, seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer.batch_encode_plus(
        sequence_examples,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=1024,
    )

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)

    return emb_0_per_protein


if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # pre-downloaded model
    tokenizer = T5Tokenizer.from_pretrained('/root/autodl-tmp/cafa6/prot_t5', local_files_only=True, do_lower_case=False)
    model = T5EncoderModel.from_pretrained('/root/autodl-tmp/cafa6/prot_t5', local_files_only=True,).to(device)
    model.eval()

    kaggle_dataset = config['base_path']  # sys.argv[1]
    output_path = os.path.join(config['base_path'], config['embeds_path'], 't5')  # sys.argv[2]
    os.makedirs(output_path, exist_ok=True)

    fn = os.path.join(kaggle_dataset, 'Train', 'train_sequences.fasta')
    sequences = SeqIO.parse(fn, "fasta")
    num_sequences = sum(1 for seq in sequences)
    sequences = SeqIO.parse(fn, "fasta")

    ids = []
    embeds = np.zeros((num_sequences, 1024))

    with tqdm.tqdm(
        total=num_sequences,
        desc="Train T5 embeddings",
        unit="seq",
        dynamic_ncols=True,
    ) as pbar:
        for i, seq in enumerate(sequences):
            ids.append(seq.id)
            embeds[i] = get_embeddings(model, tokenizer, str(seq.seq)).cpu().numpy()
            pbar.update(1)

    np.save(os.path.join(output_path, 'train_embeds.npy'), embeds)
    np.save(os.path.join(output_path, 'train_ids.npy'), np.array(ids))

    fn = os.path.join(kaggle_dataset, 'Test', 'testsuperset.fasta')

    sequences = SeqIO.parse(fn, "fasta")
    num_sequences = sum(1 for seq in sequences)
    print("Number of sequences in test:", num_sequences)
    sequences = SeqIO.parse(fn, "fasta")

    ids = []
    embeds = np.zeros((num_sequences, 1024))

    with tqdm.tqdm(
        total=num_sequences,
        desc="Test T5 embeddings",
        unit="seq",
        dynamic_ncols=True,
    ) as pbar:
        for i, seq in enumerate(sequences):
            ids.append(seq.id)
            embeds[i] = get_embeddings(model, tokenizer, str(seq.seq)).cpu().numpy()
            pbar.update(1)

    np.save(os.path.join(output_path, 'test_embeds.npy'), embeds)
    np.save(os.path.join(output_path, 'test_ids.npy'), np.array(ids))
