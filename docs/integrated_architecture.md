# QPPG + EHC Integrated Architecture
## "Primitive Genesis Through Compositional Validation"

```
                    INTEGRATED ARCHITECTURE
                    =======================

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
    |     BLIND --------> Trigger bifurcation (below)    |
    +---------------------|-----------------------------+
                          |
             +------------+------------+
             |                         |
          FAMILIAR                   BLIND
             |                         |
             v                         v
    +------------------+    +------------------------+
    |   EHC SOLVER     |    |  LAYER 1: SUBSTRATE    |
    |  (Existing DSL)  |    |  (Primitive Genesis)   |
    |                  |    |                        |
    | P0: 100 base     |    | QPPGSubstrate          |
    | P1: extended     |    | - Increase mu          |
    | P2: patterns     |    | - Bifurcation occurs   |
    | Phi: compressed  |    | - New attractor forms  |
    |                  |    | - near BLIND input     |
    | search.py        |    |                        |
    | solver.py        |    | Output: unnamed        |
    | compositional_   |    | dynamical primitive    |
    |   solver.py      |    | f_new(grid) -> grid    |
    +--------|---------+    +-----------|------------+
             |                          |
             |         +----------------+
             |         |
             v         v
    +--------------------------------------------------+
    |        COMPOSITIONAL VALIDATION ENGINE             |
    |    (EHC's search.py + solver.py + QPPG primitive)  |
    |                                                    |
    |  The new QPPG primitive is injected into EHC's     |
    |  program library as an unnamed candidate:          |
    |                                                    |
    |  P_extended = P0 U P1 U P2 U {f_new}              |
    |                                                    |
    |  EHC's compositional solver tests:                 |
    |    - f_new alone on task demos                     |
    |    - f_new o rot90 o ...  (depth-2 compositions)   |
    |    - ... o f_new o ...  (as inner transform)       |
    |                                                    |
    |  Evaluation: does f_new (alone or composed)        |
    |  produce correct output for ALL demo pairs?        |
    |                                                    |
    |  Result: USEFUL / NOT_USEFUL                       |
    +-----------|---------|-----------------------------+
                |         |
           USEFUL    NOT_USEFUL
                |         |
                v         v
    +-----------------+  +-------------------------+
    |  LAYER 2:       |  |  ECL ENERGY DECAY       |
    |  CRYSTALLIZE    |  |                         |
    |                 |  |  EnergyAccountant       |
    |  1. Program     |  |  - No energy allocated  |
    |     synthesis:  |  |  - Attractor decays     |
    |     find f_sym  |  |  - Forgotten under      |
    |     that matches|  |    pressure             |
    |     f_new's I/O |  |  (Gate 4, rho=0.964)   |
    |                 |  +-------------------------+
    |  2. LLM names   |
    |     the new     |
    |     primitive:  |
    |     "diagonal   |
    |      mirror +   |
    |      flood fill"|
    |                 |
    |  3. ECL energy  |
    |     allocated   |
    |     to maintain |
    |     new attractor|
    |                 |
    |  4. Added to    |
    |     DSL as P_new|
    |     = named,    |
    |     tested,     |
    |     energy-     |
    |     budgeted    |
    |     primitive   |
    +-----------------+
                |
                v
    +--------------------------------------------------+
    |              GROWING PRIMITIVE LIBRARY              |
    |                                                    |
    |  P0 (static):  rotate, flip, recolor, ...         |
    |  P1 (static):  segment, per_object, flood_fill    |
    |  P2 (static):  tile, pattern_complete, ...        |
    |  Phi (learned): compressed from solutions          |
    |  P_qppg (new): attractor-derived, validated,      |
    |                named, energy-budgeted              |
    |                                                    |
    |  The library GROWS when blindness triggers         |
    |  genesis and compositional testing validates.      |
    |  It SHRINKS when energy pressure forces            |
    |  consolidation of unused primitives.               |
    +--------------------------------------------------+


    COMPLETE LOOP (per task):
    =========================

    1. Task arrives (input/output grid pairs)
    2. Layer 3: Embed task, check blindness
       - FAMILIAR -> EHC solves with existing library
       - BLIND -> trigger primitive genesis
    3. Layer 1: QPPG creates new attractor (dynamical primitive)
    4. Compositional Validation: EHC tests new primitive
       - Alone and in composition with existing primitives
       - Against task demonstrations
    5a. USEFUL -> Layer 2 crystallizes:
        - Program synthesis extracts symbolic form
        - LLM provides natural language description
        - ECL allocates energy to maintain attractor
        - New primitive joins library
    5b. NOT_USEFUL -> ECL lets attractor decay
    6. System moves to next task with (possibly) expanded library


    VALIDATION STATUS:
    ==================

    Component             | Status    | Evidence
    ----------------------|-----------|---------------------------
    QPPG Substrate        | VALIDATED | ARI=0.997 (d=8, exp5)
    Blindness Detection   | VALIDATED | P=1.00, R=0.83 (exp12)
    Energy Budget         | VALIDATED | rho=1.00 (exp12,13)
    EHC Solver            | VALIDATED | 37/400 ARC tasks
    EHC DSL (100+ prims)  | VALIDATED | 54% improvement
    LLM Description       | VALIDATED | R^8 domain (exp10)
    Trust Scoring         | VALIDATED | rho=0.70 GSM8K (exp11)
    ----------------------|-----------|---------------------------
    QPPG as grid cluster  | FAILED    | ARI~0.00 (exp13b)
    Hybrid SUCN+LLM       | FAILED    | ARI<0.20 (exp13b)
    Blindness on grids    | FAILED    | Recall=0.07 (exp13b)
    ----------------------|-----------|---------------------------
    QPPG->EHC integration | NOT BUILT | This is the next step
    Attractor as primitive | NOT BUILT | Needs decoder design
    Compositional testing  | NOT BUILT | Needs primitive injection
    Energy-gated library   | NOT BUILT | Needs integration wiring


    KEY INSIGHT: Why exp13/13b failed but the architecture can work
    ================================================================

    exp13/13b asked: "Can QPPG CLUSTER grid transforms?"
    Answer: No. The feature space is too overlapping.

    The architecture asks: "Can QPPG CREATE NEW grid transforms?"
    This is a fundamentally different question.

    QPPG doesn't need to CLASSIFY existing transforms.
    It needs to GENERATE new dynamical processes that,
    when composed with EHC's existing primitives,
    solve tasks that neither could solve alone.

    The substrate creates attractors in grid-embedding space.
    Each attractor IS a transformation (input -> flow -> output).
    Quality is tested by EHC's solver, not by ARI clustering.
```
