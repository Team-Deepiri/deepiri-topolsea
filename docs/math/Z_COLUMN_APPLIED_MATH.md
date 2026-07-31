# Z-Column — Applied Mathematics Discovery

Following `exovra-jayden` applied-math skill: observation before formalism.
**Scope:** the shipping fractal index (`dv-index-zcolumn`), not DESIGN_PLAN Langlands / p-adic / ghost (deferred).

---

## System (plain language, no specialized vocabulary)

We store many numbered lists of numbers (embeddings). To find the lists most similar to a new list, we do not want to compare against every stored list every time.

Topolsea’s novel idea: squash each list into a point on a square, then put that point into a box that sits inside a larger box that sits inside a still larger box — like nested picture frames. All lists that land in the same box are stacked on top of each other. To search, start at some frame, look at nearby stacks, and if that is not enough, climb back out and try a neighboring stack. Afterwards, re-check the shortlist with the original full-precision distance.

What actually ships today also has a second path that, when unsure, fans out across many boxes until nearly every list has been touched. That second path is where the “sublinear search” claim lives or dies.

---

## Inventory

### Entities
- **Vector** — length-`d` float list with an id
- **Projected point** — two numbers in `[0,1)` from a fixed random ± projection + tanh map
- **Cell** — labeled box `(layer, x, y)`
- **Column stack** — ordered bag of ids in one cell, plus a running average (centroid), quantized copies, and a hot/cold access weight
- **Fractal grid** — nested layers; each deeper layer covers a smaller centered square with a smaller grid
- **Query trail** — which cells were visited, how many times search “backed up,” how many candidates were scored

### Actions (discrete events)
- Insert: project → deepest cell → push onto that column; update centroid; store full vector
- Delete: remove id from maps and from its column
- Search: choose entry layer → beam over columns → optional climb-back → neighborhood rings → ranked-column fallback → exact rerank of shortlist
- Record access / rebalance: decay hotness; sometimes promote/demote vectors between layers
- Persist / rebuild: rewrite blobs; if index blob missing, rebuild from vectors

### Measurable quantities (with units)

| Quantity | Unit | Notes |
|---|---|---|
| `d` | dim | embedding length |
| `N` | count | corpus size |
| `G₀` | cells | outer grid width (default 16) |
| `L` | layers | max_layers (default 3) |
| `ρ` | 1 | pitch_ratio (default 0.5) |
| `ef` | count | beam / search budget |
| `k` | count | top-k |
| `C_scan` | columns/query | columns_scanned |
| `V_touch` | vectors/query | candidate_pool |
| `R` | 1 | revert_count |
| recall@k | 1 | overlap vs flat GT |
| `t_p50` | ms | latency |
| `Q` | 1/s | QPS |
| memory | bytes | serialized / quantized / FP32 |

### Constraints / always-true
- Projection seed fixed ⇒ same vector always maps to same cell (until config changes)
- Each live id appears in `vectors` map exactly once
- On insert, id goes to **exactly one** deepest cell (assignment is a partition of the corpus)
- Metric is one of {L2, cosine, dot}; dimension must match
- Nested layers: deeper extent = `ρ^ℓ` of the unit square, inset toward center

---

## Representations examined

### Diagram (nested addressing)

```
layer 0: full [0,1)² tiled G₀×G₀
   └─ layer 1: centered ρ×ρ square, finer cells
        └─ layer 2: centered ρ²×ρ² square
insert: v ↦ (p_x,p_y) ↦ deepest cell containing point
search: entry layer → child descent → sibling revert → ring expand → centroid-ranked columns → FP32 rerank
```

### Time series (boring region)
Under default search, `revert_count` stays ~0 across queries while `candidate_pool` saturates at `N`. The “boring” constant is not the fractal address — it is **fallback always finishing the job**. That is the skeleton of today’s runtime behavior.

### Hand-worked example
Outer 8×8, `L=3`, `ρ=0.5`:

| layer | width | origin | extent |
|------:|------:|-------:|-------:|
| 0 | 8 | 0.00 | 1.00 |
| 1 | 4 | 0.25 | 0.50 |
| 2 | 2 | 0.375 | 0.25 |

- Center `(0.5,0.5)` → deepest `(2,1,1)`
- Corner `(0.05,0.05)` → deepest `(0,0,0)` only (outside inner squares)

**Conserved check:** insert 4 distinct vectors → `|vectors|=4` and sum of column heights = 4. Delete one → both drop by 1. Assignment partition holds.

---

## Candidate invariants

| Kind | Candidate | Status after break attempts |
|---|---|---|
| Conserved | Σ column heights = `N` | **Holds** (insert/delete/tests). Compaction promote currently can **duplicate** an id into another column without removing it from source — **breaks** partition if hot promote fires. |
| Conserved | Each id ↔ one deepest cell under fixed projection | **Holds** at insert time. Compaction promote/demote can move copies — invariant becomes “primary address vs working copies.” |
| Bounded | Projected coords in `[0, 0.9999]` | **Holds** by clamp |
| Monotone | Access EMA decays with elapsed time | **Holds** for unused columns |
| Structural | Nested-square containment: deeper cell ⇒ point in all outer layers’ extents | **Holds** by construction of `FractalGrid` |
| Claimed | Callback-reverse fires often enough to matter | **Fails empirically** at 10k default: avg `revert_count = 0` |
| Claimed | Search touches ≪ `N` vectors | **Fails** at default: `V_touch / N = 1.0` |

**Exchange pattern:** when “columns scanned” stays modest but `V_touch→N`, the conserved “work” is not columns — it is **vector distance evaluations**. Measuring only columns hides exhaustive vector scan inside fat columns / wide beams.

---

## Candidate symmetries

| Symmetry | Holds? | Consequence |
|---|---|---|
| Relabel vector ids | Yes | Aggregate column stats OK |
| Rotate embedding basis | **No** — signed projection axes are fixed by seed | Shard keys / cells are **not** rotation-invariant; changing embedding space requires rebuild |
| Scale embeddings (cosine) | Approximately if norms fixed upstream | Unit-normalize in benches |
| Scale fractal pitch `ρ` | Changes nest geometry | Must re-index |
| Observer: linear vs log coords on square | Description symmetry of `[0,1)²` only after tanh | Absolute projection pre-tanh is not meaningful; cell membership is |
| Compositional (add shard) | Hash(column_key) % shards | Relabeling shards ok; beam routing must include neighbors |

**Ruled out by symmetry:** any claim that “same semantic neighborhood ⇒ same cell under arbitrary orthogonal transforms” without fixing the projection.

---

## Dimensionless groups

Relevant quantities: `N, d, G₀, L, ρ, ef, k, V_touch, C_scan, t, recall`.

Independent combinatorial units: **vector-count**, **dim**, **cell-count**, **time**.

Useful groups:

1. **Touch fraction** `τ = V_touch / N` — must be ≪ 1 for sublinear claim  
2. **Recall ratio** `ρ_r = recall_Z / recall_HNSW` — go/no-go ≥ 0.98  
3. **Latency ratio** `λ = t_Z / t_HNSW` — go/no-go ≤ 1.5  
4. **Beam load** `β = ef / k`  
5. **Grid capacity** `Γ = Σ_ℓ (G₀ ρ^ℓ)²` (upper bound on addressable cells)  
6. **Fill** `φ = N / (# nonempty columns)` — mean column height  

**Limiting cases**
- `ef → ∞` or rings → cover grid ⇒ `τ → 1`, recall → 1, latency → flat-like  
- `ef → 1`, rings = 0, fcols = 0 ⇒ pure beam; recall drops (measured ~0.82 at 10k)  
- `N → 1` ⇒ all methods equal  
- `ρ → 1` ⇒ layers coincide; fractal nesting vanishes  

Unknown function of interest: `recall ≈ F(τ, β, φ)` with `F` increasing in `τ` — today’s high recall is mostly high `τ`, not clever nesting.

---

## Candidate state variables

| Candidate | Markov? | Notes |
|---|---|---|
| Full HNSW graph | Yes (approx) | Baseline |
| Set of nonempty columns + centroids + vectors | Yes for exact search | What Z-Column stores |
| Access EMA per column | Needs clock | Summary of history for compaction |
| Predictor weights + layer hit EMA | Online SGD | History summary; **polluted** because `used_fallback_scan` is always set true before observe |

**Minimal useful reduced state for routing:** `(projection_seed, grid params, column_key → shard)`.  
**Minimal useful reduced state for *fast* ANN:** not yet achieved — search state effectively expands to near-full corpus via beam+fallback.

---

## Is this optimization?

Yes, at query time:

- **Decision:** which columns (then vectors) to score  
- **Objective (soft):** maximize recall@k subject to latency / touch budget  
- **Hard constraints:** dimension match; return ≤ k  
- **Information:** query vector + index; no future queries  

Protocol go/no-go is exactly this multi-objective: recall parity **and** latency/touch budgets. Satisfying only recall by raising `τ` is optimizing the wrong single objective.

---

## Conceptual model (category first)

**Category:** deterministic combinatorial partition of `R^d` via a fixed Johnson–Lindenstrauss-style 2-row sign projection composed with a nested rectangular partition of `[0,1)²`, plus a beam + backtrack walk on the induced column graph, plus exact rerank.

**Not:** O(1) Langlands operator, p-adic memory, or symplectic ghost dynamics (those remain research fiction relative to the code).

### Incremental assembly (what is real)

1. **Addressing map** `A(v) = deepest_cell(Π(v))` — sound, measurable, used for insert + shard keys.  
2. **Column centroid index** — coarse filter; approximate.  
3. **Revert beam** — intended correction for greedy miss; **empirically idle** under current budgets.  
4. **Neighborhood + ranked fallback** — safety net; **empirically does the recall work** and destroys sublinearity.  
5. **Hybrid FP32 rerank** — standard; fine.  
6. **Compaction by hotness** — optional; can break unique assignment if promote duplicates.

### Refined falsifiable formulas (no free magic constants)

Let `S` be the set of columns the searcher opens. Approximate cost:

\[
\mathrm{Cost} \approx \alpha\,|S| + \gamma \sum_{c\in S} h(c)
\]

with `h(c)` = column height, `γ` = cost of one distance (exact path today).  
Touch fraction:

\[
\tau \approx \frac{1}{N}\Big|\bigcup_{c\in S}\mathrm{ids}(c)\Big|
\]

**Prediction P1.** If `|S|` covers a constant fraction of nonempty columns under default rings/`ef`, then `τ → 1` as columns fill evenly.  
**Prediction P2.** Holding `τ` fixed (cap rings + fcols + effective beam), recall@10 of Z-Column vs HNSW falls below 0.98 before latency ratio hits 1.5 — i.e. the fractal walk alone is not yet HNSW-competitive.  
**Prediction P3.** `revert_count` stays near 0 whenever fallbacks run unconditionally after the beam.

---

## Proof strategy / plausibility

Claim: “Z-Column is a different species that beats HNSW on density/explainability/recovery without exhaustive scan.”

- Small example: addressing nest works (hand grid).  
- Limiting case: uncapped fallback ⇒ exhaustive (observed).  
- Adversarial: disable fallback ⇒ recall gap opens (observed 0.82 vs HNSW 0.95).  

**Verdict:** addressing + explainability + DR rebuild are real. ANN *efficiency* claim is **not** supported at 10k/128d under protocol gates.

---

## Simplifications made

| Simplification | Cost |
|---|---|
| Ignore Langlands/p-adic/ghost | Focus on shippable math |
| Synthetic unit-sphere data | May differ from clustered text embeddings |
| Serialize footprint ≠ RSS | Buyer TCO approximate |
| Cap fix for `max_fallback_columns` | Wired; neighborhood rings still dominate |

---

## Experiments before solutions

Harness: `topolsea-math-probe`, `topolsea-prove`.  
See `docs/math/EXPERIMENT_RESULTS.md`.

Also fixed measurement bug: `max_fallback_columns` was configured but unused; now enforced in ranked fallback.

---

## Domain of validity — where this should fail

- High `ef` + large ring radius on small `G₀` ⇒ exhaustive scan  
- Embedding distribution far from projection’s concentration ⇒ empty / overloaded cells  
- Compaction promote under load ⇒ duplicate membership / recall weirdness  
- Concurrent mutation during search without locks on maps (Phase A)  
- Filtered queries (post-filter) — outside this math scope but product-critical  

---

## Failed guesses and what each revealed

| Guess | Failure | Lesson |
|---|---|---|
| Callback-reverse is the active differentiator | `revert_avg=0` at 10k | Safety nets short-circuit the novel path |
| `max_fallback_columns=96` bounds work | Neighborhood rings still touch ~all vectors | Bound **vector** evaluations, not only ranked columns |
| High recall ⇒ good index | Bought by `τ=1` | Optimize multi-objective; don’t celebrate single metric |
| Predictor learns entry layer | Always sees fallback flag | Fix observe signal before claiming learning |

---

## Generalization check

Fractal **partition keys** generalize cleanly to sharding (M4). Fractal **search** generalizes only if beam+fallback are rewritten so `τ(N)` grows slower than linear with a proven recall lower bound — open problem, not a packaging issue.

---

## Immediate math → eng priorities (before Phase A polish)

1. Make fallback conditional; measure revert rate when it matters  
2. Bound `V_touch` explicitly (hard budget) and expose `τ` in explain  
3. Coarse quantized scan in beam (`coarse_only=true`) before FP32  
4. Fix compaction promote to move, not copy  
5. Only then claim go/no-go vs HNSW on equal-memory **bounded-touch** regimes  
