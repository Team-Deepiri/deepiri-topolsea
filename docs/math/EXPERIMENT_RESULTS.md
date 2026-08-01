# Experiment results — Z-Column applied-math campaign

Updated: 2026-07-31 (v2 after structural search fixes).  
Harness: `cargo run -p dv-bench --release --bin topolsea-math-probe -- --json`  
Raw JSON: `/tmp/topolsea-math-probe-full.json` (local run artifact).  
Math framing: [`Z_COLUMN_APPLIED_MATH.md`](Z_COLUMN_APPLIED_MATH.md). Roadmap: [`../NEXT_STEPS.md`](../NEXT_STEPS.md).

---

## Why this campaign (propel the goal)

Production goal needs **G1∧G2∧G3** (recall ≥0.98×HNSW, p50 ≤1.5×HNSW, touch τ&lt;0.5).  
Prior runs looked like “recall OK, slow.” That was **misleading**: search was accidentally near-exhaustive. This round **removes the cheats**, measures the real fractal walk, and names the engineering that unlocks a path to all three gates.

---

## Bugs found by measurement (fixed on this branch)

| Bug | Effect | Fix |
|---|---|---|
| `fallback_beam_radius.max(1)` + `max_fallback_columns.max(1)` | “rings=0,fcols=0” still ran fallback | Skip neighborhood/ranked unless caps &gt; 0; set `used_fallback_scan` only when used |
| `ef = ef.max(coarse_pool)` with `coarse_pool ≈ N/20` | Beam width ignored caller `ef`; τ→1 by construction | Keep `coarse_pool` for **rerank heap only**; beam uses caller/`ef_search` |

These are exactly the applied-math failure mode: **observer/description artifacts leaking into dynamics**.

---

## Gates (protocol)

| Gate | Pass |
|---|---|
| G1 | `recall_Z / recall_HNSW ≥ 0.98` @ k=10 |
| G2 | `p50_Z / p50_HNSW ≤ 1.5` |
| G3 | `candidates_touched / N &lt; 0.5` |

**Result after fixes: no regime in the grid passed G1∧G2∧G3.**

---

## Experiment matrix

- Scales: N ∈ {2k, 10k, 50k}, dim=128, k=10, cosine, seed=42  
- Distros: unit sphere; 32 Gaussian clusters (σ=0.08) — held-out structure  
- Sweep: ef ∈ {8…128}, rings/beam/fcols from pure beam → default-like  
- Extra: projection-seed sensitivity at N=2k pure beam  

---

## Headline numbers (sphere, N=10 000)

HNSW baseline: recall@10 ≈ **0.955**, p50 ≈ **1.71 ms** (ef=128).

| Regime | recall | vs HNSW | p50 ms | lat× | τ touch | G123 |
|---|---:|---:|---:|---:|---:|---|
| Pure beam (rings=0,fcols=0), any ef | ~0.008 | ~0.008 | ~0.05 | ~0.03 | ~0.006 | n**YY** |
| Light fallback ef=16 rings=1 fcols=8 | 0.863 | 0.903 | 8.3 | 4.9 | 0.93 | nnn |
| Light fallback ef=24 rings=1 fcols=16 | 0.950 | 0.995 | 9.0 | 5.2 | 0.98 | **Y**nn |
| rings=2 fcols=32 | 1.000 | 1.05 | ~9 | ~5 | 1.00 | **Y**nn |
| Default-like rings=8 fcols=96 | 1.000 | 1.05 | ~9.5 | ~5 | 1.00 | **Y**nn |

**Cliff, not curve:** pure beam is fast+sparse but recall≈0; one ring of neighborhood restores recall and **collapses to τ≈1**. There is no intermediate τ with G1 under current primitives.

### Dimensionless groups (sphere N=10k, pure beam)

| Group | Value | Reading |
|---|---|---|
| β = ef/k | 0.8–12.8 | **Does not move τ or recall** once beam is honest — layer has few columns |
| τ = V_touch/N | ~0.005 | Sublinear ✓ |
| ρ_r = recall_Z/recall_HNSW | ~0.008 | Catastrophic ✗ |
| λ = p50_Z/p50_HNSW | ~0.03 | Fast ✓ |
| φ = N / nonempty_cols | ~333 | Mass piled into ~30 columns — projection is peaked |

---

## Scale check (sphere, pure beam vs light fallback)

| N | Pure beam recall | Pure τ | Light (r1,f16) recall | Light τ | Light lat× |
|---:|---:|---:|---:|---:|---:|
| 2 000 | ~0.005 | ~0.007 | ~1.00 | ~1.0 | ~2.5 |
| 10 000 | ~0.008 | ~0.006 | ~0.95 | ~0.98 | ~5 |
| 50 000 | ~0.010 | ~0.007 | ~0.95 | ~0.99 | ~16 |

As N grows, **fallback latency× worsens** while pure-beam recall stays near zero. Exhaustive-ish neighborhood does not scale.

---

## Clustered distro (important for RAG)

Absolute recall@10 vs flat GT (not vs broken HNSW ratios):

| N | HNSW (default) | Z pure beam | Z light fallback |
|---:|---:|---:|---:|
| 2 000 | 0.25 | ~0.85–1.00* | ~1.00 |
| 10 000 | 0.27 | 0.89 (pre-fix*) / ~0 after honest pure | ~1.00 |
| 50 000 | 0.25 | ~0.005 pure | ~0.99–1.00 |

\*Pre-fix pure beam still had accidental fallback. After fix, pure beam ≈0 on clusters at 50k too.

**Note:** default HNSW is a weak baseline on this clustered generator (efConstruction/M defaults). Still: Z-Column **with fallback** matches flat GT on clusters; the open problem remains **bounded τ**, not absolute recall.

On clusters at N=2k pure beam after fix, revert_frac ≈ 0.08 — first time revert is non-zero (still rare).

---

## Projection seed (description symmetry)

N=2k, ef=32, pure beam: seeds {1,42,999} → recall 0.0125 / 0.0025 / 0.0025, nonempty cols 21–24.  
**Local search is seed-sensitive**; any “learned predictor” on top of a dead beam will not save ANN quality.

---

## Failed guesses → what to build (Track M → goal)

| Guess | Failure | Next build (propels G1∧G2∧G3) |
|---|---|---|
| Fractal beam alone is ANN | recall ~1% at τ~0.5% | Better expand — but see Phase-2 oracle |
| Raising `ef` trades τ for recall | ef inert once coarse_pool decoupled; few columns/layer | Need inter-column edges **and** intra-column prune |
| Neighborhood rings = controlled expand | First ring ⇒ τ→1 | Hard `V_touch` budget; do not use rings as the retriever |
| Fallback is a safety net | Fallback **is** the retriever | Budgeted (M2) + rare (M1) |
| **Centroid-kNN / M-graph alone** | Even **oracle** column pick needs B≈6–8 at τ≈0.68 for G1 | **M3+M4 first** (partial column scan + height balance); M-graph closes centroid→oracle gap |
| Predictor will fix entry layer | Beam path too narrow; observe signal was polluted | Fix partition + prune + graph first |

### Recommended Track M sequence (updated after Phase-2)

1. **M0 (done):** honest caps; stop inflating ef with coarse_pool  
2. **M3:** quantized coarse scan inside columns; FP32 only at rerank *(unblocks τ while visiting oracle’s ~8 cols)*  
3. **M4:** compaction / split so hot columns do not own most of N  
4. **M-graph:** neighbor list among nonempty column centroids (close gap to oracle picker)  
5. **M2:** hard `V_touch` / distance-eval budget in explain + early exit  
6. **M1:** ring/ranked fallback **only** if heap&lt;k or score gap  
7. **M5:** re-probe with `topolsea-math-localize` + `topolsea-math-probe`; publish only if G1∧G2∧G3  

Until that lands: **default product ANN = HNSW**; Z-Column = explain + fractal shard keys (still valuable).

---

## v1 snapshot (before honesty fixes) — do not market

Default config looked like recall≥HNSW at ~5× latency with τ=1. That was **exhaustive fallback**, not fractal skill. Retained only as a cautionary baseline in git history / earlier notes.

---

## Domain of validity

- Synthetic unit sphere + simple Gaussian clusters ≠ production text embeddings  
- HNSW not retuned per distro  
- Latency is single-threaded release binary on one machine  
- Column graph redesign not yet implemented — conclusions about **current** code, not the ceiling of the addressing idea  

---

## Stopping condition for this loop

Surprises stopped: every new scale repeats the same cliff (fast/empty vs slow/exhaustive).  
Ready to **commit engineering** (centroid graph + budgets) rather than more parameter sweeps of the same operator.

---

## Phase-2: localization + centroid-kNN + oracle (M-graph stress test)

Harness: `cargo run -p dv-bench --release --bin topolsea-math-localize -- --n=10000`  
(also ran N∈{2k,10k,50k}; 40 queries; G1 = recall≥0.98 **vs flat**).

### What we measured

1. **Where GT lives** — for each query, which fractal columns hold the true top-10.  
2. **Centroid-kNN expand (`centB*`)** — scan the B columns whose centroids are nearest the query (online IVF / naive M-graph).  
3. **Oracle column expand (`orclB*`)** — cheat: pick the B columns that actually contain the most GT mass. Upper bound on **any** whole-column picker.

### Headline (sphere, N=10 000, φ≈333, 30 nonempty cols)

| Fact | Value |
|---|---|
| GT in nearest-centroid cell | **5%** |
| Unique columns housing one query's GT@10 | p50=**6**, p99=**8** (oracle floor on B) |
| `centB8` recall / τ | 0.44 / 0.28 |
| `orclB8` recall / τ / lat× | **1.00 / 0.68 / 2.45** → G123 = **Ynn** |
| `orclB4` recall / τ | 0.79 / 0.50 → still &lt;G1 |
| Any method with G1∧G2∧G3 | **none** |

Same story at N=50k (φ≈1515): oracle floor still p50≈6 columns; `orclB8` hits recall 1.0 at τ≈0.68. Centroid-kNN recall matches the localization CDF exactly (scoring is consistent; mass just is not in the nearest centroids).

### Falsification (important)

**Whole-column expand cannot reach G1∧G3 under the current partition**, even with an oracle column picker.  
Reason: neighbors are scattered across ~6–8 columns, and those columns are the **tall** ones — scanning them already touches ≳65% of N.  
So “add a centroid graph and walk B≪#cols” is **necessary for better column selection** (centroid leaves ~2× recall on the table vs oracle at B=8) but **not sufficient** for the product gates.

### Revised Track M (what to build next)

| Priority | Work | Why |
|---|---|---|
| **M3** | Quantized / residual **intra-column** prune; FP32 only on survivors | Only way to visit ~8 GT columns at τ&lt;0.5 |
| **M4** | Height-balance / split hot columns (move-not-copy) | Shrink φ so oracle’s B columns are not most of the corpus |
| **M-graph** | Centroid neighbor graph (close centroid→oracle gap) | Still needed so online search tracks the oracle curve |
| M2 / M1 | Hard `V_touch` budget; conditional fallback | Keep τ honest once prune+graph exist |
| M5 | Re-run localize + math-probe | Publish only on G1∧G2∧G3 |

Until then: **product ANN = HNSW**; Z-Column = explain + fractal shard keys.
