# Allopp-MPL

**Allopp-MPL** infers allopolyploid species networks from rooted gene trees, accounting for incomplete lineage sorting. It maximizes a rooted-triplet composite (pseudo)likelihood in which the gene copies of each polyploid species are assigned to its subgenomes, then fits branch lengths in coalescent units. The number of reticulations is supplied by the user.

- Input: rooted gene trees (one per locus) and a map from species to their gene copies
- Output: the best-scoring network (extended Newick) and a plain-text run report
- Supports mixed ploidy: diploids, tetraploids, hexaploids and higher, in one analysis

## Install

Allopp-MPL requires Python 3.12 or newer. Installing its PhyNetPy dependency also needs Git and a C compiler. PhyNetPy is pinned to commit `b3a1a411bb32a86f2136285c0fd928719be97533` and is installed automatically by `pip install .`.

```bash
git clone https://github.com/Moerz/Allopp-MPL.git
cd Allopp-MPL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Check the installation:

```bash
allopp-mpl --version
# Allopp-MPL 0.1.0
```

To run the test suite:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Quick start

Allopp-MPL needs two input files.

**Gene trees** (`gene_trees.nwk`): one rooted, binary Newick tree per locus, one per line. Leaves are labelled with gene-copy names. Branch lengths and internal labels are allowed, but they are ignored.

```text
((a,x1),(b,x2));
((a,x2),(b,x1));
```

**Gene map** (`gene_map.json`): each species and the labels of its gene copies. The number of copies is the species' number of subgenomes; for example, a tetraploid has two.

```json
{
  "A": ["a"],
  "B": ["b"],
  "X": ["x1", "x2"]
}
```

Every leaf label in the trees must appear exactly once in the map. The map must list every subgenome of each species, even if some loci are missing copies.

Run inference with one reticulation on the bundled example:

```bash
allopp-mpl infer \
  --gene-trees examples/gene_trees.nwk \
  --gene-map examples/gene_map.json \
  --reticulations 1 \
  --output example-results/
```

```text
Best score -5.545177444479562 from stream 1; wrote example-results/network.newick and example-results/results.txt
```

The output directory contains:

| File | Content |
|---|---|
| `network.newick` | Best network found, in extended Newick, with branch lengths in coalescent units |
| `results.txt` | Best score, the outcome and seed of every search stream, and warnings |

The output directory must not already exist. Allopp-MPL never overwrites earlier results.

A typical analysis of a real dataset uses a larger search budget:

```bash
allopp-mpl infer \
  --gene-trees gene_trees.nwk \
  --gene-map gene_map.json \
  --reticulations 3 \
  --evaluations 5000 \
  --streams 3 \
  --output results/
```

## Options

`allopp-mpl infer [options]`

| Option | Default | Description |
|---|---|---|
| `--gene-trees FILE` | required | Rooted gene trees, one Newick tree per line |
| `--gene-map FILE` | required | JSON map from species to gene-copy labels |
| `--reticulations N` | required | Number of reticulations in the inferred network. Allopp-MPL does not choose this number. |
| `--output DIR` | required | New directory for the results |
| `--evaluations N` | 2048 | Search budget per stream: the number of candidate networks fitted |
| `--streams N` | 3 | Number of independent searches. The best-scoring result is reported. |
| `--starts N` | 4 | Starting networks screened at the beginning of each stream. At most four go on to full search, and the budget then goes adaptively to the most promising one. |
| `--seed N` | 17 | Random seed. Stream 1 uses this seed; the seeds of later streams are derived from it and recorded in `results.txt`. |
| `--start-network FILE` | none | Start every stream from this network (extended Newick, species-labelled leaves, exactly `--reticulations` reticulations, branch lengths in coalescent units) instead of automatically built starts |
| `--max-init-states N` | 5000 | Limit on the construction search for each automatically built start. Hitting it is reported as a construction failure, not as evidence that the number of reticulations is impossible. |
| `--initializer NAME` | `ploidy-compatible` | How starting networks are built (`additive-v1` is the same method under its original name) |
| `--outgroup LABEL` | none | Gene-copy label used to root the trees, then removed. Omit it if the trees are already rooted without an outgroup. |

**Tips**

- **Choosing the number of reticulations.** To compare several numbers, run `infer` once per number. The scores are composite likelihoods, so comparing them directly is not a calibrated test for the number of reticulations.
- **Parallel runs.** Streams run one after another. To use several cores, run separate `--streams 1` jobs with different `--seed` values, then keep the highest score.
- **Reading the score.** Higher scores are better for the same input. Agreement across streams indicates a stable search; it is not a statistical confidence measure.

## License

Allopp-MPL is released under the [MIT License](LICENSE). It depends on [PhyNetPy](https://github.com/NakhlehLab/PhyNetPy), which is also MIT-licensed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

