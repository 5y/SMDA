"""Explicitly create a new disjoint SFT set, recording every dropped pair."""
import argparse
from pathlib import Path
from smda.data import read_rows, validate_pairs, prompt_id, write_json


def run(sft_data, eval_data, output, *, drop=False):
    pairs=validate_pairs(read_rows(sft_data))
    evaluation={prompt_id(r['prompt']) for r in read_rows(eval_data)}
    overlap=[r for r in pairs if prompt_id(r['prompt']) in evaluation]
    if overlap and not drop:
        raise ValueError(f'{len(overlap)} SFT/evaluation overlaps. Use --drop-overlapping-sft only for an explicitly new experiment.')
    kept=[r for r in pairs if prompt_id(r['prompt']) not in evaluation]
    validate_pairs(kept,[r['prompt'] for r in read_rows(eval_data)])
    write_json(output,kept)
    write_json(str(output)+'.manifest.json',{'status':'new disjoint experiment; not historical 200-pair replication',
        'original_n':len(pairs),'retained_n':len(kept),'removed_pair_ids':[r['pair_id'] for r in overlap]})
    return len(kept)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sft-data',required=True);p.add_argument('--eval-data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--drop-overlapping-sft',action='store_true')
    a=p.parse_args()
    print(run(a.sft_data,a.eval_data,a.output,drop=a.drop_overlapping_sft),'retained pairs')
