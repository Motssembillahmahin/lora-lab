# Experiment Log

One entry per training run. Each entry is tied to a git commit SHA so any result
can be reproduced by checking out that commit's config + code.

Template:

```
## Run NNN — <short name>
- commit: <git sha>
- hypothesis: <what this run tests, one variable changed>
- config: <the diff from the previous run>
- final loss: <eyeballed trend>
- wall time: <minutes>
- trainable params: <count / total>
- observation: <what happened, surprises>
- next: <what this suggests trying>
```

---

<!-- Add Run 001 after the first `make train`. -->
