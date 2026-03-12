# Quantum-Principled Primitive Genesis: A Dynamical Systems Architecture for Program Synthesis and Adaptive Problem Solving

**Avighan Majumder**

---

## Abstract

We present Quantum-Principled Primitive Genesis (QPPG), a three-layer architecture that autonomously expands a program solver's primitive library through bifurcation-driven attractor creation, compositional validation, and energy-constrained lifecycle management. The architecture addresses a fundamental limitation of enumerative program synthesis: fixed primitive libraries cannot solve tasks requiring transformations outside their design vocabulary. QPPG integrates three independently validated components — (1) a bifurcation substrate that creates new computational attractors via parameter-driven phase transitions, (2) an LLM-guided semantic encoding and synthesis pipeline, and (3) a metacognitive blindness detection and energy budgeting system — into a closed loop where failure triggers genesis, compositional testing validates utility, and energy pressure ensures library parsimony.

We validate the architecture through 14 experiments spanning synthetic clustering (ARI = 0.997), trust calibration (AUROC = 0.686 on GSM8K), blindness detection (precision = 1.00, recall = 0.83, AUROC = 0.9996), and a full integration test on 400 real ARC-AGI training tasks. The integration experiment demonstrates that QPPG-guided primitive synthesis solves 3 previously unsolvable ARC tasks (7.1% gain over baseline) with zero regressions, though we identify that the bifurcation substrate currently contributes clustering via KMeans fallback rather than its native dynamical integration on grid-domain embeddings. We discuss honest limitations, the architectural pivot from classification to generation, and pathways toward harder AGI benchmarks.

---

## 1. Introduction

### 1.1 The Fixed-Library Problem

Enumerative program synthesis approaches like DreamCoder (Ellis et al., 2021), ARC solvers (Chollet, 2019), and domain-specific language (DSL) search engines share a fundamental constraint: their primitive libraries are fixed at design time. When a task requires a transformation outside the library's expressive reach, no amount of compositional search will find a solution. The solver is *blind* to the task — it lacks the computational vocabulary to even express the answer.

This paper introduces an architecture that addresses this limitation through a biologically-inspired loop:

1. **Detect blindness**: Recognize when current primitives cannot solve a task
2. **Generate candidates**: Create new computational primitives through guided synthesis
3. **Validate compositionally**: Test candidates alone and in composition with existing primitives
4. **Budget energy**: Maintain useful primitives, decay useless ones

### 1.2 The QPPG Architecture

QPPG draws on three theoretical principles:

- **Bifurcation theory**: New computational categories emerge through parameter-driven phase transitions in a dynamical energy landscape, analogous to how new stable states appear in physical systems at critical points
- **Compositional validation**: A new primitive has value only if it solves tasks (alone or composed with existing primitives) that were previously unsolvable
- **Energy-constrained growth**: A finite energy budget forces the system to prioritize useful primitives and shed unused ones, preventing unbounded library growth

The architecture operates as three layers:

- **Layer 1 (Substrate)**: A differentiable energy landscape where points flow to attractors. Bifurcation creates new attractors; each attractor basin represents a computational category
- **Layer 2 (Encoding + Synthesis)**: LLM-based semantic description of input/output pairs, followed by embedding and program synthesis
- **Layer 3 (Metacognition)**: Blindness detection (five-signal weighted classifier) and energy accounting (budget-constrained attractor maintenance)

### 1.3 Key Architectural Insight

The original QPPG design used the bifurcation substrate as a **classifier** — clustering existing transforms into categories. Experiments 13 and 13b demonstrated this fails on grid-domain embeddings (ARI ≈ 0.00). The architectural pivot reframes the substrate as a **generator**: it doesn't need to classify existing transforms; it needs to identify *failure patterns* that guide the creation of *new* transforms. Quality is validated by the compositional solver, not by clustering accuracy.

---

## 2. Architecture

### 2.1 System Overview

```
    INPUT: ARC Task (input/output grid pairs)
    ==========================================
                      |
                      v
    +--------------------------------------------------+
    |              LAYER 3: METACOGNITION               |
    |            (Blindness Detection + Trigger)         |
    |                                                    |
    |   BlindnessDetector                                |
    |   - 5-signal weighted probability                  |
    |   - DBSCAN noise, residual energy, distance,       |
    |     velocity, convergence                          |
    |                                                    |
    |   Classification:                                  |
    |     FAMILIAR -----> Use existing primitives        |
    |     BLIND --------> Trigger genesis (below)        |
    +---------------------|-----------------------------+
                          |
             +------------+------------+
             |                         |
          FAMILIAR                   BLIND
             |                         |
             v                         v
    +------------------+    +------------------------+
    |   EHC SOLVER     |    |  FAILURE CLUSTERING    |
    |  (Existing DSL)  |    |  + PRIMITIVE GENESIS   |
    |                  |    |                        |
    | P0: 208 base     |    | 1. Embed failures via  |
    | P1: extended     |    |    LLM descriptions    |
    | P2: patterns     |    | 2. Cluster failures    |
    | Phi: compressed  |    | 3. Synthesize new      |
    |                  |    |    primitives per       |
    | search.py        |    |    cluster              |
    | solver.py        |    | 4. Validate on demos   |
    +--------|---------+    +-----------|------------+
             |                          |
             |         +----------------+
             |         |
             v         v
    +--------------------------------------------------+
    |        COMPOSITIONAL VALIDATION ENGINE             |
    |                                                    |
    |  New primitives injected into EHC library:         |
    |  P_extended = P0 U P1 U P2 U {f_new}              |
    |                                                    |
    |  Depth-1: f_new alone on task demos                |
    |  Depth-2: f_new o rot90 o ... (compositions)      |
    |                                                    |
    |  Result: USEFUL / NOT_USEFUL                       |
    +-----------|---------|-----------------------------+
                |         |
           USEFUL    NOT_USEFUL
                |         |
                v         v
    +-----------------+  +-------------------------+
    |  CRYSTALLIZE    |  |  ENERGY DECAY            |
    |                 |  |                          |
    |  1. Joins DSL   |  |  EnergyAccountant        |
    |  2. Gets energy |  |  - No energy allocated   |
    |     allocation  |  |  - Attractor decays      |
    |  3. Compositions|  |  - Shed under pressure   |
    |     auto-built  |  |                          |
    +-----------------+  +--------------------------+
```

### 2.2 Component Interfaces

| Component | Interface | Validated |
|-----------|-----------|-----------|
| `QPPGSubstrate` | `integrate(x) -> x_final`, `find_attractors(x) -> labels` | Exp 5: ARI=0.997 |
| `QPPGOnlineClusterer` | `process_all(points) -> labels` | Exp 5: d=8-16 |
| `BlindnessDetector` | `detect(points) -> {blind_indices, signals}` | Exp 12: AUROC=0.9996 |
| `EnergyAccountant` | `tick() -> state`, `consolidate() -> new_centers` | Exp 4: rho=0.964 |
| `LLMEncoder` | `encode_with_descriptions(in, out) -> (emb, desc)` | Exp 10: R^8 domain |
| `EHCSolver` | `solve(task) -> {solved, program_name}` | Exp 14: 42/400 ARC |

### 2.3 The Compositional Validation Engine

A critical architectural decision: new primitives are **not** evaluated in isolation. Instead, they are injected into the EHC solver's library and tested through the full compositional search pipeline:

1. **Library injection**: `solver.library['new_prim'] = fn`
2. **Program rebuilding**: `build_depth1_programs(library)` creates single-primitive programs; `build_depth2_programs(library)` creates all valid two-primitive compositions
3. **Search**: The constraint-extracting search engine tests each program against all demonstration pairs
4. **Result**: A primitive is "useful" if it (alone or in composition) perfectly solves at least one previously-unsolvable task

This design means a primitive with only partial correctness (e.g., 3/9 demos correct on representative tasks) can still solve specific tasks perfectly when composed with geometric, color, or structural transforms.

---

## 3. Experimental Validation

We conducted 14 experiments organized as a gated validation pipeline. Each gate tests a specific architectural claim.

### 3.1 Experiment Summary Table

| Exp | Component | Domain | Key Metric | Value | Gate | Verdict |
|-----|-----------|--------|------------|-------|------|---------|
| 4 | Energy Budget | Synthetic | Spearman rho (budget vs attractors) | 0.964 | >0.95 | **PASS** |
| 5 | Substrate Clustering | Synthetic d=16 | ARI (5 clusters) | 0.997 | >0.95 | **PASS** |
| 11 | Trust Scoring | GSM8K | AUROC | 0.686 | Beat baselines | **PASS** |
| 12 | Blindness Detection | Synthetic | Precision / Recall / AUROC | 1.00 / 0.83 / 0.9996 | P>0.95, AUROC>0.99 | **PASS** |
| 13 | Grid Clustering | ARC transforms | QPPG ARI (train) | 0.004 | >=0.15 | **FAIL** |
| 13b | Hybrid Grid | ARC transforms | QPPG ARI (hybrid) | 0.001 | >=0.30 | **FAIL** |
| **14** | **Full Integration** | **400 ARC tasks** | **Newly solved tasks** | **3 (+7.1%)** | **>=1** | **PASS** |

### 3.2 Experiment 5: Substrate Validation (PASS)

**Setup**: 5-cluster synthetic data in 16 dimensions, 200 points per phase, 5 seeds.

**Results**: QPPG achieves ARI = 0.997, far exceeding DBSCAN (0.536), Growing-KMeans (0.252), and Fixed-KMeans(3) (0.104). The substrate correctly discovers the number of clusters through bifurcation without being told k.

**Significance**: Validates the core bifurcation mechanism — the substrate can discover categorical structure through parameter sweeps rather than explicit k specification.

### 3.3 Experiment 12: Blindness Detection (PASS)

**Setup**: 5 attractor basins, 100 test trajectories per seed (5 seeds). Tests whether the detector can identify inputs that fall outside all known basins.

**Results**:
- Precision: 1.00 (no false alarms — never incorrectly flags familiar inputs)
- Recall: 0.826 (catches 83% of truly blind inputs)
- AUROC: 0.9996 (near-perfect discrimination)
- Individual signal AUROCs: Distance = 1.0, Velocity = 0.986, Energy = 0.867, Noise = 0.908

**Significance**: The five-signal weighted classifier provides reliable metacognitive awareness. The system knows what it doesn't know.

### 3.4 Experiment 11: Trust Scoring (PASS)

**Setup**: 200 GSM8K-style math problems, 10 candidate solutions per problem, 3 seeds.

**Results**: QPPG trust score AUROC = 0.686 (±0.016), outperforming Majority Fraction baseline (0.627), Embedding Tightness (0.575), and Bifurcation Score (0.500). Spearman correlation rho = 0.304 (p < 0.001).

**Significance**: The energy landscape provides a meaningful quality signal for solution verification — deeper attractor wells correlate with correct solutions.

### 3.5 Experiments 13/13b: Grid Domain Failure (FAIL)

**Setup**: 600 ARC grid transforms, LLM and SUCN feature extraction, QPPG clustering.

**Results**: QPPG ARI ≈ 0.000 across all conditions. Post-integration spread collapses from 1.000 to 0.022 — all points converge to a single attractor.

**Root Cause Analysis**: Grid-domain LLM embeddings produce heavily overlapping features (KMeans ARI only 0.07-0.19). When basin separation is insufficient, the integration dynamics correctly report "no cluster structure" by converging everything to one attractor. This is honest behavior, not a bug — QPPG refuses to force artificial clusters.

**Architectural Response**: This failure motivated the pivot from QPPG-as-classifier to QPPG-as-generator, leading to the Gate 6 integration design.

### 3.6 Experiment 14: Gate 6 Integration (PASS)

This is the central experiment of this paper. It tests the full architecture loop on 400 real ARC-AGI training tasks.

#### Part A: EHC Baseline

The EHC solver (v3 configuration with depth-2 compositions, extended primitives, pattern primitives, constraints, invariants, episodic memory, compression operator, demo-adaptive programs, and guess-and-check) solves **42/400 tasks (10.5%)** in 20 seconds using 1,770 programs (208 base, 1,560 depth-2, 2 phi-compressed).

#### Part B: Failure Embedding and Clustering

358 failed tasks were embedded via Claude Haiku descriptions (1,167 API calls, $1.11). Each task's demo pairs were described in natural language, embedded via sentence-transformers (all-MiniLM-L6-v2, 384-d), averaged across demos to produce task-level embeddings, then PCA-reduced to 16-d (63.7% variance explained) and standardized.

**QPPG clustering**: 1 cluster (silhouette = -1.0). Confirms exp13b finding.

**KMeans fallback**: k = 13 clusters (silhouette = 0.106). Used as active clustering for downstream primitive synthesis.

**Blindness detection**: 0/358 blind (all tasks classified as familiar). Expected since all are from the same ARC training distribution — blindness detection is designed for out-of-distribution inputs.

#### Part C: Primitive Genesis

Claude Sonnet synthesized primitives for each of the 13 clusters (2 attempts each with self-correction). Results:

| Cluster | Tasks | Primitive | Demos Correct | Status |
|---------|-------|-----------|---------------|--------|
| 0 | 20 | qppg_cluster_0 | 1/9 | Generated |
| 1 | 13 | qppg_cluster_1 | 3/9 | Generated |
| 2 | 23 | qppg_cluster_2 | 3/9 | Generated |
| 3 | 25 | - | 0/0 | Failed (syntax) |
| 4 | 35 | - | 0/9 | Failed |
| 5 | 32 | qppg_cluster_5 | 1/8 | Generated |
| 6 | 68 | - | 0/9 | Failed |
| 7 | 28 | - | 0/9 | Failed |
| 8 | 14 | - | 0/8 | Failed |
| 9 | 30 | - | 0/9 | Failed |
| 10 | 20 | qppg_cluster_10 | 2/9 | Generated |
| 11 | 31 | - | 0/8 | Failed |
| 12 | 19 | - | 0/9 | Failed |

5 primitives generated from 13 clusters. Notably, even primitives with partial validation accuracy (e.g., 3/9) proved useful because the compositional search engine discovers specific task-primitive pairings.

#### Part D: Re-run with Expanded Library

**QPPG-guided condition**: 5 synthesized primitives injected into EHC library, programs rebuilt with new depth-2 compositions.

**Results**:
- **3 newly solved tasks** (7.1% gain over baseline):
  - `1fad071e` — solved by `qppg_cluster_2` (pattern restoration)
  - `3428a4f5` — solved by `qppg_cluster_1` (section comparison)
  - `ce4f8723` — solved by `qppg_cluster_1` (section comparison)
- **0 regressions** on 42 originally-solved tasks (confirmed via controlled comparison: fresh baseline 42/42, fresh+injection 42/42)

**Random control**: 5 primitives synthesized from randomly-grouped failed tasks (no clustering). Also solved 3 tasks. This tie (QPPG-guided = random) is discussed in Section 5.

#### Part E: Energy Budget

Energy accountant tracked primitive lifecycle over 5 cycles:
- 2 primitives useful (`qppg_cluster_1`: 2 uses, `qppg_cluster_2`: 1 use)
- 3 primitives useless (correctly identified for decay)
- Budget pressure reaches consolidation threshold at cycle 0 (13 clusters expensive)
- 4 + 5 attractors shed across cycles 3-4

#### Gate Verdicts

| Gate | Criterion | Result |
|------|-----------|--------|
| B1 | Failure clusters > 1 | **PASS** (13 via KMeans) |
| B2 | Silhouette > 0 | **PASS** (0.106) |
| D1 | >=1 task newly solved | **PASS** (3 tasks) |
| D2 | Gain >= 5% over baseline | **PASS** (7.1%) |
| D3 | 0 regressions | **PASS** (0) |
| D4 | QPPG-guided >= random | **PASS** (3 >= 3, tied) |

**Final Verdict: PASS** (all 6 gates satisfied)

---

## 4. Results Audit and Validity Assessment

We present an honest assessment of the experimental evidence, including concerns and limitations.

### 4.1 What Is Genuinely Validated

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Bifurcation substrate discovers cluster structure | ARI = 0.997 on synthetic data (exp5, 5 seeds) | **HIGH** |
| Energy budget constrains library growth | rho = 0.964 budget-attractor correlation (exp4) | **HIGH** |
| Blindness detection identifies OOD inputs | AUROC = 0.9996, P = 1.00, R = 0.83 (exp12, 5 seeds) | **HIGH** |
| Trust scoring calibrates solution quality | AUROC = 0.686 on GSM8K (exp11, 3 seeds) | **MODERATE** |
| Full loop solves new ARC tasks | +3 tasks, 0 regressions (exp14) | **MODERATE** |
| Primitives compose correctly with existing DSL | depth-2 programs auto-generated and tested (exp14) | **HIGH** |

### 4.2 Honest Concerns and Limitations

#### 4.2.1 QPPG Substrate Does Not Contribute to Grid Clustering

The most significant finding is negative: the QPPG bifurcation substrate contributes **nothing** to grid-domain clustering. In exp14, QPPG found 1 cluster while KMeans found 13. The architecture's Gate B1 (clusters > 1) passes only because of the KMeans fallback.

**Root cause**: Grid-domain LLM embeddings produce overlapping features. The substrate's integration dynamics correctly converge all points to a single attractor when basin boundaries are insufficiently separated. This is fundamentally different from the synthetic domain (exp5) where clusters are well-separated in feature space.

**Implication**: The QPPG substrate's value in the current architecture is limited to domains with clear feature separation. The grid-domain results rely on standard clustering (KMeans) rather than bifurcation-based category discovery.

#### 4.2.2 Gate D4 Is a Tie, Not a Win

QPPG-guided synthesis solved exactly as many tasks (3) as the random control (3). Gate D4 passes on a `>=` condition, but the clustering provided no measurable advantage over random task grouping for primitive synthesis. This suggests that:

1. The LLM's synthesis capability is the primary driver of new task solutions
2. The clustering adds organizational structure but doesn't improve synthesis quality
3. A larger-scale experiment might reveal a difference, but n=3 vs n=3 is inconclusive

#### 4.2.3 Small Effect Size

3 newly solved tasks out of 358 failures is a 0.84% solve rate on previously-unsolvable tasks. While this exceeds the gate threshold (>5% relative gain), the absolute improvement is modest. The primitives solved only tasks where the LLM happened to generate code matching a specific task's pattern.

#### 4.2.4 Primitive Quality Is Low

Only 5/13 clusters produced compilable, partially-correct primitives. Of those 5, only 2 proved useful in compositional search. The 38% synthesis success rate and 40% utility rate suggest significant room for improvement in the LLM synthesis pipeline.

#### 4.2.5 Blindness Detection Not Useful on In-Distribution Data

In exp14, blindness detection classified 0/358 failed tasks as blind (all familiar). This is correct behavior — all tasks come from the same ARC training distribution — but it means blindness detection didn't contribute to the integration. Its value would emerge with genuinely novel task distributions.

### 4.3 Statistical Validity

| Aspect | Assessment |
|--------|------------|
| Exp 5 (substrate) | 5 seeds, consistent ARI > 0.99 across all | **Strong** |
| Exp 12 (blindness) | 5 seeds, tight confidence intervals | **Strong** |
| Exp 11 (trust) | 3 seeds, moderate variance | **Adequate** |
| Exp 14 (integration) | Single run, no seed variation | **Weak** |
| Exp 14 Gate D4 | n=3 vs n=3, no statistical power | **Inconclusive** |

The integration experiment (exp14) represents a single run. The 42-task baseline and 3-task improvement have not been replicated across seeds. While the regression check validates that results are not artifacts, the small sample size limits generalizability claims.

---

## 5. Figures

### Figure 1: Failure Cluster t-SNE Visualization

![Cluster t-SNE](../results/exp14_gate6/plots/cluster_tsne.png)

*t-SNE projection of 358 failed ARC task embeddings (384-d LLM descriptions, averaged across demo pairs). Colors indicate KMeans cluster assignments (k=13, silhouette=0.106). Clusters show moderate separation, with some overlap consistent with the low silhouette score. QPPG substrate collapsed all points to a single attractor on this data.*

### Figure 2: Solve Rate Comparison

![Solve Rate](../results/exp14_gate6/plots/solve_rate_comparison.png)

*ARC task solve rates across three conditions: EHC baseline (42 tasks), EHC + QPPG-guided primitives (45 tasks), EHC + random-control primitives (45 tasks). Both experimental conditions improve over baseline by 3 tasks (7.1%), with no statistical difference between cluster-guided and random synthesis.*

### Figure 3: Energy Budget Dynamics

![Energy Budget](../results/exp14_gate6/plots/energy_budget.png)

*Left: Remaining energy over 5 simulation cycles. Budget depletes rapidly due to high attractor count (13 clusters). Right: Budget pressure reaches consolidation threshold (0.8) immediately, triggering attractor shedding. 9 attractors shed across cycles 3-4, demonstrating the energy system's ability to prune unused computational categories.*

### Figure 4: Experimental Validation Pipeline

```
    Exp 4          Exp 5          Exp 11         Exp 12
    Energy         Substrate      Trust          Blindness
    Budget         Clustering     Scoring        Detection
    rho=0.964      ARI=0.997      AUROC=0.686    AUROC=0.9996
    PASS           PASS           PASS           PASS
      |              |              |              |
      +--------------+--------------+--------------+
                                |
                    Exp 13/13b: Grid Domain
                    QPPG ARI = 0.000
                    FAIL (clustering)
                    PASS (energy only)
                                |
                    Architectural Pivot:
                    Classifier -> Generator
                                |
                    Exp 14: Full Integration
                    +3 tasks, 0 regressions
                    PASS (all 6 gates)
```

### Figure 5: Primitive Lifecycle

```
    Cluster 0-12 (from KMeans on failure embeddings)
         |
    LLM Primitive Synthesis (Claude Sonnet)
         |
    +----+----+----+----+----+----+----+----+----+----+----+----+----+
    | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |C10 |C11 |C12 |
    +----+----+----+----+----+----+----+----+----+----+----+----+----+
    | 1/9| 3/9| 3/9| ERR| 0/9| 1/8| 0/9| 0/9| 0/8| 0/9| 2/9| 0/8| 0/9|
    +----+----+----+----+----+----+----+----+----+----+----+----+----+
      OK   OK   OK  FAIL FAIL  OK  FAIL FAIL FAIL FAIL  OK  FAIL FAIL
         |    |    |                  |                  |
    Injected into EHC solver library
         |    |    |                  |                  |
    Compositional search (depth-1 + depth-2)
         |    |    |                  |                  |
         0    2    1                  0                  0    <- tasks solved
              |    |
         USEFUL  USEFUL -> Energy maintained
                                     |                  |
                               USELESS x3 -> Energy decayed, shed
```

---

## 6. Discussion

### 6.1 The Architecture Works, QPPG Substrate Does Not (On Grids)

The most important takeaway from this work is a decomposition result: the *architecture* (detect failure → cluster failures → synthesize primitives → validate compositionally → energy-budget) demonstrably adds value (+3 ARC tasks, +7.1%). However, the QPPG bifurcation substrate — the theoretical centerpiece — does not contribute to the grid domain.

The substrate excels in synthetic domains with clean feature separation (ARI = 0.997). Grid-domain LLM embeddings violate this assumption. The 384-d sentence-transformer vectors for grid transform descriptions occupy a highly overlapping region of embedding space, and PCA reduction to 16-d does not create the basin boundaries that QPPG's integration dynamics require.

### 6.2 LLM Synthesis Is the Real Value-Add

The experiment reveals that Claude Sonnet's ability to synthesize `transform(grid) -> grid` functions from example task demonstrations is the primary source of new solve capability. The clustering (whether QPPG or KMeans) provides organizational structure but doesn't measurably improve synthesis quality — random grouping achieves equal results.

This suggests the optimal architecture may be simpler than proposed: for each failed task, directly prompt an LLM to synthesize a candidate primitive, validate it compositionally, and retain if useful. The clustering adds value primarily for reducing the number of LLM calls (one per cluster rather than one per failed task).

### 6.3 Compositionality Amplifies Weak Primitives

A subtle but important finding: primitives with only 33% demo accuracy (3/9) on representative tasks nevertheless solved specific tasks perfectly. This happens because compositional search discovers specific primitive-transform pairings that the validation sampling missed. For example, `qppg_cluster_1` (a section-comparison primitive) solved 2 tasks that required comparing grid halves separated by a divider — a pattern the primitive captured correctly even though its general accuracy was low.

This validates the architectural decision to test primitives through EHC's full compositional search rather than filtering by standalone accuracy alone.

### 6.4 Energy Budget as Library Hygiene

The energy budget system correctly identified 3/5 generated primitives as useless and began shedding them. While the simulation is simplified (5 cycles, predetermined usage counts), it demonstrates the principle: finite energy prevents unbounded library growth. Over multiple task-solving episodes, useful primitives accumulate energy (deeper wells) while useless ones decay.

---

## 7. Pathways to Harder AGI Problems

### 7.1 Current Limitations for AGI-Scale Problems

The architecture as implemented faces several challenges when scaling to harder problems:

1. **Sequential API calls**: 1,167 Haiku calls took 39 minutes. Scaling to 10,000+ tasks requires parallelization
2. **Shallow composition depth**: Depth-2 compositions (f(g(x))) are insufficient for tasks requiring 3+ step reasoning
3. **Task-specific synthesis**: LLM primitives tend to be task-specific rather than truly reusable abstractions
4. **No incremental learning**: Each primitive synthesis is independent — the system doesn't learn from which synthesis strategies succeed

### 7.2 Proposed Modifications for AGI-Hard Benchmarks

#### 7.2.1 Hierarchical Primitive Abstraction

Instead of synthesizing flat primitives, the architecture should support hierarchical composition:

```
Level 0: Base primitives (rotate, recolor, crop)
Level 1: Synthesized primitives (section_compare, pattern_restore)
Level 2: Meta-primitives composed of Level 1 primitives
Level 3: Strategy-level programs
```

This mirrors DreamCoder's library learning but with the energy budget providing principled pruning.

#### 7.2.2 Cross-Task Transfer via Embedding Similarity

Currently, primitives are synthesized per-cluster. A stronger approach:

1. When a new task fails, find the nearest-neighbor solved task in embedding space
2. Retrieve the solving program and attempt to adapt it (rather than synthesizing from scratch)
3. Use the QPPG substrate to detect when the adaptation creates a genuinely new computational category (bifurcation) vs. a minor variant (same basin)

#### 7.2.3 Active Learning for Primitive Quality

Replace the current 2-attempt synthesis with an active learning loop:

1. Synthesize candidate primitive
2. Test on cluster tasks, identify failure cases
3. Feed failure cases back to LLM with error analysis
4. Iterate until primitive achieves >80% demo accuracy or budget is exhausted
5. Track which error patterns lead to successful synthesis (meta-learning)

#### 7.2.4 Improved Embedding for QPPG Substrate

The substrate's failure on grid embeddings stems from feature quality, not substrate mechanics. Potential improvements:

- **Contrastive fine-tuning**: Train the sentence-transformer to produce more separable embeddings for transform descriptions (using exp13's labeled data as supervision)
- **Multi-modal embedding**: Combine grid pixel features with LLM descriptions through learned fusion
- **Structured embeddings**: Instead of averaging demo-pair embeddings, use a relational network that captures inter-pair structure

#### 7.2.5 Scaling to ARC-AGI-2 and Beyond

For harder benchmarks (ARC-AGI-2, BIG-Bench Hard, mathematical reasoning):

1. **Deeper search**: Extend from depth-2 to depth-4 compositions with type-guided pruning
2. **Iterative refinement**: Run the full loop multiple times (solve → fail → synthesize → re-solve → fail on harder subset → synthesize more)
3. **Multi-domain transfer**: Train the substrate on diverse domains (math, logic, spatial reasoning) so it develops richer attractor structure
4. **Hypothesis-driven synthesis**: Instead of clustering failures, use the blindness detector to identify *why* the system fails (missing operation type) and direct synthesis accordingly

#### 7.2.6 Toward Self-Improving Systems

The architecture's closed loop (failure → genesis → validation → retention) is a minimal form of self-improvement. To approach AGI-relevant capabilities:

1. **Reflection**: After solving new tasks, analyze *why* the new primitive worked and extract general principles
2. **Curriculum learning**: Order tasks by difficulty, building primitive libraries incrementally
3. **Meta-primitive synthesis**: Synthesize not just primitives but synthesis strategies — programs that generate programs
4. **Open-ended exploration**: Allow the system to generate and solve its own problems, expanding its library proactively rather than only reactively

---

## 8. Related Work

- **DreamCoder** (Ellis et al., 2021): Library learning through compression. QPPG extends this with energy-based pruning and LLM-guided synthesis
- **ARC-AGI** (Chollet, 2019): The benchmark driving this work. Our 42/400 baseline and +3 improvement place us in the mid-range of DSL-based approaches
- **Program synthesis via LLMs** (Chen et al., 2021; Li et al., 2023): Direct LLM code generation for tasks. Our contribution is the compositional validation loop that discovers value in imperfect synthesized code
- **Bifurcation detection** (Strogatz, 2015): QPPG's substrate draws on nonlinear dynamics for category discovery
- **Energy-based models** (LeCun, 2006): The energy landscape framework provides a natural substrate for attractor-based computation
- **Metacognition in AI** (Cleeremans, 2011): Blindness detection implements a simple form of metacognitive awareness

---

## 9. Conclusion

QPPG demonstrates a viable architecture for autonomously expanding a program solver's capabilities through failure-driven primitive genesis. The key contributions are:

1. **A closed-loop architecture** connecting failure detection, guided synthesis, compositional validation, and energy-constrained library management
2. **Empirical validation** on 400 real ARC tasks showing the loop solves 3 previously-unsolvable tasks with zero regressions
3. **Honest failure analysis**: The bifurcation substrate does not contribute to grid-domain clustering; the architecture succeeds despite this, primarily through LLM synthesis and compositional search
4. **A clear path forward**: Hierarchical abstraction, active learning, and improved embeddings could amplify the architecture's effectiveness on harder benchmarks

The architecture's value lies not in any single component but in how the components compose: imperfect clustering guides imperfect synthesis, which produces imperfect primitives, which the compositional search engine nevertheless discovers specific perfect applications for. This tolerance for imperfection at each layer, combined with strict end-to-end validation (does the task actually get solved?), is what makes the loop work.

---

## Appendix A: Reproducibility

All code and results are available in the QPPG repository:

- Substrate: `qppg/core.py` (QPPGSubstrate, QPPGOnlineClusterer)
- Encoder: `qppg/encoder.py` (LLMEncoder, LLMTransformDescriber)
- Blindness: `qppg/blindness.py` (BlindnessDetector, EnergyAccountant)
- Integration: `experiments/exp14_gate6_integration.py`
- Results: `results/exp14_gate6/`
- EHC Solver: `ehc_solver/core/` (solver.py, primitives.py, search.py)

**Runtime**: 48.2 minutes on M-series MacBook.
**API cost**: $1.11 (embedding) + ~$0.50 (synthesis) = ~$1.61 total.
**Dependencies**: Python 3.12, numpy, scipy, scikit-learn, sentence-transformers, anthropic, torch.

## Appendix B: ARC Tasks Newly Solved

| Task ID | Primitive Used | Pattern |
|---------|---------------|---------|
| `1fad071e` | `qppg_cluster_2` | Pattern restoration in repeating grids |
| `3428a4f5` | `qppg_cluster_1` | Section comparison across grid dividers |
| `ce4f8723` | `qppg_cluster_1` | Section comparison with difference mapping |

## Appendix C: Complete Gate Criteria

| Gate | Definition | Threshold | Measured | Pass |
|------|-----------|-----------|----------|------|
| B1 | Failure clusters discovered | > 1 | 13 | Yes |
| B2 | Cluster quality | silhouette > 0 | 0.106 | Yes |
| D1 | Absolute improvement | >= 1 newly solved | 3 | Yes |
| D2 | Relative improvement | >= 5% gain | 7.1% | Yes |
| D3 | Safety (no regressions) | 0 regressions | 0 | Yes |
| D4 | Clustering adds value | guided >= random | 3 >= 3 | Yes (tie) |
