# The corpus

The text the released models were trained on, so this repository can be cloned and
trained from without running the generators. It is generated, not hand-written -
`corpus/` holds the programs that produce it, and `corpus/chat/conversations.txt` holds
the only part a person wrote by hand.

| | training.txt | heldout.txt |
|---|---|---|
| characters | 30,000,724 | 1,801,888 |
| conversations | 20,874 | 1,230 |
| turns | 388,970 | 23,268 |
| model turns that think first | 193,223 of 194,485 | 11,597 of 11,634 |
| distinct characters | 82 | 82 |

```
training.txt  sha256 1eb88a0e8d8b0a36e485228035190f785743ad6d46946e50fe080bd6e97535c4
heldout.txt   sha256 9d95c5bde34b6f6d2e97f6e135bb0f710b27b4fb45305d8fa28e6ba057d1c0cf
```

## Train on it

```bash
py train.py --data data/training.txt --val_data data/heldout.txt \
    --preset large --block_size 384 --fresh --dropout 0.0 --steps 32000 --select_by train
```

Held-out conversations never appear in training, so the validation loss means
something. Both files use the turn markers described in
[../docs/spec.md](../docs/spec.md).

## Regenerate it

```bash
py corpus/chat/make_corpus.py --target_chars 20000000 --composed 0.95 --think 1.0
py tools/publish_corpus.py
```

The generators are seeded, so the same command on the same commit reproduces these
files byte for byte - the hashes above are worth checking if you change anything in
`corpus/` and want to know whether you changed the corpus too.
