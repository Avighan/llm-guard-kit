"""
Example: Trust verification for LLM math solutions.

Demonstrates TrustScorer on pre-generated solutions.
Requires: ANTHROPIC_API_KEY env var (for SolutionGenerator).

For a quick demo without API access, this example includes
pre-generated mock solutions to show the scoring pipeline.
"""

import numpy as np
from qppg import TrustScorer, extract_numerical_answer

# --- Demo with mock solutions (no API needed) ---

# Simulate a problem where the LLM is consistent (should be high trust)
consistent_solutions = [
    "Let's solve step by step. 3 * 4 = 12, then 12 + 5 = 17. #### 17",
    "First multiply: 3 times 4 equals 12. Add 5: 12 + 5 = 17. #### 17",
    "3 * 4 = 12. 12 + 5 = 17. The answer is #### 17",
    "Multiplication first: 3*4=12. Addition: 12+5=17. #### 17",
    "Step 1: 3 x 4 = 12. Step 2: 12 + 5 = 17. #### 17",
]

# Simulate a problem where the LLM is inconsistent (should be low trust)
inconsistent_solutions = [
    "3 + 4 = 7, then 7 * 5 = 35. #### 35",
    "3 * 4 = 12, plus 5 = 17. #### 17",
    "First 4 + 5 = 9, then 3 * 9 = 27. #### 27",
    "3 * (4 + 5) = 3 * 9 = 27. #### 27",
    "3 * 4 + 5 = 12 + 5 = 17. #### 17",
]

scorer = TrustScorer()

print("=== Consistent Solutions ===")
result = scorer.compute_trust(consistent_solutions)
print(f"  Trust score: {result['qppg_trust']:.3f}")
print(f"  Majority answer: {result['majority_answer']} ({result['majority_fraction']:.0%})")
print(f"  Embedding tightness: {result['embedding_tightness']:.3f}")
print(f"  QPPG clusters: {result['n_clusters_qppg']}")

print("\n=== Inconsistent Solutions ===")
result = scorer.compute_trust(inconsistent_solutions)
print(f"  Trust score: {result['qppg_trust']:.3f}")
print(f"  Majority answer: {result['majority_answer']} ({result['majority_fraction']:.0%})")
print(f"  Embedding tightness: {result['embedding_tightness']:.3f}")
print(f"  QPPG clusters: {result['n_clusters_qppg']}")

# --- Full pipeline with live API (requires ANTHROPIC_API_KEY) ---
# Uncomment below to run with real API calls:
#
# from qppg import SolutionGenerator
#
# generator = SolutionGenerator(temperature=0.8)
# problem = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. She sells the rest at the farmers' market for $2 each. How much does she make every day?"
#
# solutions = generator.generate(problem, n=10, seed=42)
# answers = [extract_numerical_answer(s) for s in solutions]
# trust = scorer.compute_trust(solutions, answers)
#
# print(f"\nLive trust: {trust['qppg_trust']:.3f}")
# print(f"Answer: {trust['majority_answer']} (${trust['majority_answer']}/day)")
# print(f"Cost: {generator.get_cost_summary()}")
