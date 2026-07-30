# The corpus

The text the released models were trained on, so this repository can be cloned and
trained from without running the generators. It is generated, not hand-written -
`corpus/` holds the programs that produce it, and `corpus/conversations.txt` holds
the only part a person wrote by hand.

| | training.txt | heldout.txt |
|---|---|---|
| characters | 20,000,072 | 1,200,010 |
| conversations | 99,061 | 6,056 |
| turns | 317,152 | 19,004 |
| model turns that think first | 149,176 of 158,576 | 9,037 of 9,502 |
| distinct characters | 80 | 80 |

```
training.txt  sha256 6f9e60ccb9997c6c8ae17e1d0be303f91a3363a44c765b2c0ff4706da7d763e0
heldout.txt   sha256 9eb9e9f66509f010f031c79d28cefb6dd705df72a377f0e4e616b7168bbe35c8
```

## Train on it

```bash
py train.py --data data/training.txt --val_data data/heldout.txt \
    --preset large --block_size 384 --fresh --dropout 0.0 --steps 20000 --select_by train
```

Held-out conversations never appear in training, so the validation loss means
something. Both files use the turn markers described in
[../docs/spec.md](../docs/spec.md).

## Regenerate it

```bash
py corpus/make_corpus.py --target_chars 20000000 --composed 0.95 --think 1.0
py tools/publish_corpus.py
```

The generators are seeded, so the same command on the same commit reproduces these
files byte for byte - the hashes above are worth checking if you change anything in
`corpus/` and want to know whether you changed the corpus too.

Models trained on this corpus: medium_think_1.2.
