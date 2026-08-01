# The corpus

The text the released models were trained on, so this repository can be cloned and
trained from without running the generators. It is generated, not hand-written -
`corpus/` holds the programs that produce it, and `corpus/chat/conversations.txt` holds
the only part a person wrote by hand.

| | training.txt | heldout.txt |
|---|---|---|
| characters | 30,001,835 | 1,801,756 |
| conversations | 21,063 | 1,291 |
| turns | 393,302 | 23,806 |
| model turns that think first | 195,416 of 196,651 | 11,836 of 11,903 |
| distinct characters | 80 | 80 |

```
training.txt  sha256 e005337891013103d9e2ff5ef5a9509706d961166e54e03bc142b7cb82650520
heldout.txt   sha256 b3137d7c3da71dfa5d9cbd6bb5200673598149affeed9305340d68f2753670a0
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
py corpus/chat/make_corpus.py --target_chars 30000000 --composed 0.97 --think 1.0
py tools/publish_corpus.py
```

The generators are seeded, so the same command on the same commit reproduces these
files byte for byte - the hashes above are worth checking if you change anything in
`corpus/` and want to know whether you changed the corpus too.

Models trained on this corpus: small_1.4, medium_1.3, medium_think_1.3, large_1.2.
