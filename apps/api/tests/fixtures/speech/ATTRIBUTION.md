# Speech fixtures

Short excerpts of the **FLEURS** corpus (Conneau et al., 2022; https://huggingface.co/datasets/google/fleurs),
licensed **CC-BY-4.0**. Real human speakers reading Wikipedia-derived sentences, in Portuguese (Brazil) and
English (US). Each file joins two utterances with a short pause and is re-encoded as Opus (32 kbps, 16 kHz, mono).
`references.json` holds the exact reference transcript of every file; the regression tests measure the
recogniser's Word Error Rate against it.
