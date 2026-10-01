# Head rankings shipped with the paper

One directory per model. `pointer_cattok.csv` ranks all heads by attention-mass R@1 on ICIR-MCQ
(6,000 samples, cattok position); `writer_cattok_proj_mean_fullpool.csv` ranks them by the R@1 of the
trained per-head image retriever (writer criterion, 1,200 held-out samples). Column `r1` / `r1_dedup`
is the number used in the paper.

| model | pointer top-1 | R@1 | writer top-1 | R@1 | pointer-writer overlap @100 |
|---|---|---|---|---|---|
| InternVL3.5-8B | L21H10 | 0.890 | L24H23 | 0.489 | 47 |
| InternVL3.5-4B | L21H10 | 0.866 | L21H31 | 0.472 | 42 |
| Qwen3-VL-8B | L21H10 | 0.922 | L24H23 | 0.595 | 35 |
| Qwen3-VL-4B | L21H10 | 0.905 | L24H23 | 0.509 | 43 |
