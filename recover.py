import codecs
import os
import json
import sys
import re


if __name__ == '__main__':
    pred_path = sys.argv[1]
    tgt_path = pred_path.replace('output.txt', 'hypothesis.txt')
    assert pred_path is not None

    aligns_path = './data/amr2.0/test.map'
    aligns = []

    with codecs.open(aligns_path, 'r') as f:
        lines = f.readlines()
    for idx in range(len(lines)):
        align = json.loads(lines[idx])
        for key in align:
            align[key] = align[key].lower()
        aligns.append(align)

    with codecs.open(pred_path, 'r') as f:
        preds = f.readlines()
    while preds[-1] == '\n':
        preds = preds[:-1]

    gens = []
    for idx in range(len(preds)):
        pred = preds[idx].strip()
        for k, v in aligns[idx].items():
            pred = pred.replace(k, v)
        gens.append(pred + '\n')

    with open(tgt_path, 'w') as f:
        f.writelines(gens)
