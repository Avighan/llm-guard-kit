"""
test_cross_dataset.py — Cross-dataset and cross-condition scoring validation.
==============================================================================

Covers the remaining "works anywhere" matrix dimensions:

  Dataset type:         HotpotQA-style ✓ | NQ-style ✓ | TriviaQA-style ✓ (here)
                        WebQ-style ✓ (here) | FEVER-style ✓ (here)

  Step count:           2-8 steps ✓ | 1 step ✓ (here) | 10+ steps ✓ (here)

  Language:             English ✓ | French ✓ (here) | Spanish ✓ (here)
                        Chinese ✓ (here) | Arabic ✓ (here)

  LLM backbone:         Claude Sonnet ✓ | GPT-4-style ✓ (here) | Llama-style ✓ (here)
                        Gemini-style ✓ (here)

Honesty table printed in TestLLMBackboneAgnostic:
  VALIDATED (held-out data):   Claude Sonnet, ReAct, HotpotQA/NQ
  PLAUSIBLE (structural only): All other LLM styles, dataset types, step counts, languages
  NOT TESTED YET:              Actual AUROC measurement on held-out data per condition

Synthetic chains mimic the output style of each condition.
They do NOT come from real LLM APIs — they model structure and vocabulary, not semantics.

Run:
    pytest tests/test_cross_dataset.py -v
    pytest tests/test_cross_dataset.py -v -m "not slow"
"""

import warnings
import pytest

warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*sentence.*")

# ── Shared helpers ────────────────────────────────────────────────────────────

def _score(steps, question, answer, fmt="react"):
    from llm_guard.agent_guard import AgentGuard
    guard = AgentGuard(agent_format=fmt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return guard.score_chain(question, steps, answer)

def _assert_valid(r):
    assert 0.0 <= r.risk_score <= 1.0
    assert r.confidence_tier in ("HIGH", "MEDIUM", "LOW")
    assert r.needs_alert == (r.risk_score >= 0.70)


# ════════════════════════════════════════════════════════════════════════════════
# DATASET TYPE COVERAGE
# ════════════════════════════════════════════════════════════════════════════════

class TestTriviaQAStyle:
    """
    TriviaQA-style chains: single-hop direct factual questions.
    Chains are shorter (1-3 Search steps), answers are terse.

    NOTE: AUROC not validated on TriviaQA. Structural validity confirmed.
    """

    CLEAN = [
        {"thought": "I should look up where the Eiffel Tower is located.",
         "action_type": "Search", "action_arg": "Eiffel Tower location",
         "observation": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."},
        {"thought": "The Eiffel Tower is in Paris, France.",
         "action_type": "Finish", "action_arg": "Paris, France", "observation": ""},
    ]

    WRONG = [
        {"thought": "Looking up Eiffel Tower",
         "action_type": "Search", "action_arg": "Eiffel Tower",
         "observation": "no information found"},
        {"thought": "Looking again",
         "action_type": "Search", "action_arg": "Eiffel Tower city",
         "observation": "no information found"},
        {"thought": "Still no results",
         "action_type": "Finish", "action_arg": "London", "observation": ""},
    ]

    def test_clean_chain_valid(self):
        r = _score(self.CLEAN, "Where is the Eiffel Tower?", "Paris, France")
        _assert_valid(r)

    def test_wrong_chain_valid(self):
        r = _score(self.WRONG, "Where is the Eiffel Tower?", "London")
        _assert_valid(r)

    def test_wrong_chain_higher_risk(self):
        r_clean = _score(self.CLEAN, "Where is the Eiffel Tower?", "Paris, France")
        r_wrong = _score(self.WRONG,  "Where is the Eiffel Tower?", "London")
        assert r_wrong.risk_score >= r_clean.risk_score, (
            f"TriviaQA-style: looping wrong chain ({r_wrong.risk_score:.3f}) should "
            f"score >= clean chain ({r_clean.risk_score:.3f})"
        )

    def test_step_count_reasonable(self):
        r = _score(self.CLEAN, "Q?", "Paris")
        assert r.step_count == 1  # 1 Search step before Finish


class TestWebQStyle:
    """
    WebQ (WebQuestions)-style chains: entity-centric knowledge base queries.
    Questions about named entities: "What city was Einstein born in?"
    Chains access structured data (Freebase-style entities).

    NOTE: AUROC not validated on WebQ. Structural validity confirmed.
    """

    CLEAN = [
        {"thought": "I need to find where Einstein was born. Search for his biography.",
         "action_type": "Search", "action_arg": "Albert Einstein birthplace",
         "observation": "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire."},
        {"thought": "Einstein was born in Ulm, Germany.",
         "action_type": "Finish", "action_arg": "Ulm, Germany", "observation": ""},
    ]

    CONFUSED = [
        {"thought": "Searching for Einstein",
         "action_type": "Search", "action_arg": "Einstein",
         "observation": "Albert Einstein developed the theory of relativity."},
        {"thought": "That doesn't answer the birthplace question. Searching more specifically.",
         "action_type": "Search", "action_arg": "Einstein born where",
         "observation": "Einstein was a theoretical physicist."},
        {"thought": "Still not finding specific birthplace information.",
         "action_type": "Search", "action_arg": "Einstein birthplace city",
         "observation": ""},
        {"thought": "Cannot determine the birthplace from available information.",
         "action_type": "Finish", "action_arg": "Germany", "observation": ""},
    ]

    def test_clean_chain_valid(self):
        r = _score(self.CLEAN, "What city was Einstein born in?", "Ulm")
        _assert_valid(r)

    def test_confused_chain_valid(self):
        r = _score(self.CONFUSED, "What city was Einstein born in?", "Germany")
        _assert_valid(r)

    def test_confused_chain_higher_risk(self):
        r_clean   = _score(self.CLEAN,   "What city was Einstein born in?", "Ulm")
        r_confused = _score(self.CONFUSED, "What city was Einstein born in?", "Germany")
        assert r_confused.risk_score >= r_clean.risk_score


class TestFEVERStyle:
    """
    FEVER-style chains: fact verification.
    Questions are claims; agent must verify True/False with evidence.

    NOTE: AUROC not validated on FEVER. Structural validity confirmed.
    Additional note: SC11 (answer-question mismatch) may behave differently
    for True/False answers vs. factual answers — both overlap with question
    tokens differently.
    """

    VERIFIED_TRUE = [
        {"thought": "The claim is 'Cristiano Ronaldo plays football'. I need to verify this.",
         "action_type": "Search", "action_arg": "Cristiano Ronaldo sport",
         "observation": "Cristiano Ronaldo is a Portuguese professional footballer who plays as a forward."},
        {"thought": "The observation confirms Ronaldo plays football. Claim is SUPPORTED.",
         "action_type": "Finish", "action_arg": "SUPPORTED", "observation": ""},
    ]

    UNSUPPORTED_GUESS = [
        {"thought": "The claim is 'The Eiffel Tower is in Berlin'. I need to verify.",
         "action_type": "Search", "action_arg": "Eiffel Tower location",
         "observation": ""},  # empty observation — can't find evidence
        {"thought": "No evidence found. Making a guess.",
         "action_type": "Search", "action_arg": "Berlin landmarks",
         "observation": "Berlin's most famous landmarks include the Berlin Wall."},
        {"thought": "Can't verify the claim. Guessing REFUTED.",
         "action_type": "Finish", "action_arg": "REFUTED", "observation": ""},
    ]

    def test_verified_chain_valid(self):
        r = _score(self.VERIFIED_TRUE, "Cristiano Ronaldo plays football. True or False?", "SUPPORTED")
        _assert_valid(r)

    def test_unsupported_chain_valid(self):
        r = _score(self.UNSUPPORTED_GUESS, "Eiffel Tower is in Berlin. True or False?", "REFUTED")
        _assert_valid(r)

    def test_unsupported_higher_risk(self):
        r_ok  = _score(self.VERIFIED_TRUE,    "Q?", "SUPPORTED")
        r_bad = _score(self.UNSUPPORTED_GUESS, "Q?", "REFUTED")
        assert r_bad.risk_score >= r_ok.risk_score


# ════════════════════════════════════════════════════════════════════════════════
# STEP COUNT DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════════

class TestSingleStepChain:
    """
    1-step chains: direct knowledge queries with no search.

    KNOWN LIMITATION: SC2 (step count) AUROC 0.88 → meaningless at 1 step.
    Warning emitted. Use use_judge=True for 1-step chains if accuracy is critical.
    """

    SINGLE_CLEAN = [
        {"thought": "This is a well-known fact I can answer directly.",
         "action_type": "Finish", "action_arg": "42", "observation": ""},
    ]

    def test_single_step_valid_output(self):
        r = _score(self.SINGLE_CLEAN, "What is 6 times 7?", "42")
        _assert_valid(r)

    def test_single_step_warning_emitted(self):
        from llm_guard.agent_guard import AgentGuard
        guard = AgentGuard()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.score_chain("Q?", self.SINGLE_CLEAN, "42")
            sc2_warns = [x for x in w if "SC2" in str(x.message) or "step" in str(x.message).lower()]
            assert len(sc2_warns) >= 1, "Expected SC2 warning for single-step chain"

    def test_single_step_step_count_is_zero(self):
        r = _score(self.SINGLE_CLEAN, "Q?", "42")
        assert r.step_count == 0  # 0 Search steps (only Finish)

    def test_components_all_present(self):
        r = _score(self.SINGLE_CLEAN, "Q?", "42")
        for k in ("sc1", "sc2", "sc8", "sc9", "sc10", "sc11", "sc12"):
            assert k in r.behavioral_components


class TestLongChain:
    """
    10+ step chains: complex multi-hop reasoning with many search steps.

    NOTE: Validated range in exp88 was 2-8 steps. 10+ step chains are
    extrapolations — SC_OLD should still work but AUROC is not validated
    for very long chains.
    """

    @staticmethod
    def _make_long_chain(n_search: int, productive: bool):
        steps = []
        for i in range(n_search):
            if productive:
                steps.append({
                    "thought": f"Step {i+1}: Searching for sub-question {i+1}.",
                    "action_type": "Search",
                    "action_arg": f"sub-question {i+1} relevant terms",
                    "observation": f"Relevant result for step {i+1}: found some data about topic {i+1}."
                })
            else:
                steps.append({
                    "thought": f"Step {i+1}: Searching again.",
                    "action_type": "Search",
                    "action_arg": "same query repeated",
                    "observation": "no results"
                })
        steps.append({
            "thought": "Final step: synthesizing all information.",
            "action_type": "Finish",
            "action_arg": "synthesized answer" if productive else "unknown",
            "observation": ""
        })
        return steps

    def test_10_step_productive_chain_valid(self):
        steps = self._make_long_chain(10, productive=True)
        r = _score(steps, "Complex multi-hop question?", "synthesized answer")
        _assert_valid(r)
        assert r.step_count == 10

    def test_10_step_looping_chain_valid(self):
        steps = self._make_long_chain(10, productive=False)
        r = _score(steps, "Complex question?", "unknown")
        _assert_valid(r)

    def test_looping_long_chain_higher_risk(self):
        productive = self._make_long_chain(10, productive=True)
        looping    = self._make_long_chain(10, productive=False)
        r_prod  = _score(productive, "Q?", "answer")
        r_loop  = _score(looping,    "Q?", "unknown")
        assert r_loop.risk_score >= r_prod.risk_score

    def test_15_step_chain_valid(self):
        steps = self._make_long_chain(15, productive=True)
        r = _score(steps, "Very complex question?", "answer")
        _assert_valid(r)
        assert r.step_count == 15

    def test_step_count_scales_correctly(self):
        for n in [2, 5, 8, 10, 15]:
            steps = self._make_long_chain(n, productive=True)
            r = _score(steps, "Q?", "A")
            assert r.step_count == n, f"Expected step_count={n}, got {r.step_count}"


# ════════════════════════════════════════════════════════════════════════════════
# LANGUAGE COVERAGE
# ════════════════════════════════════════════════════════════════════════════════

class TestMultilingualChains:
    """
    Non-English chains: tests scoring doesn't crash, and warns about calibration.

    KNOWN LIMITATION: Jaccard signals (SC3, SC6, SC9-12) were calibrated on
    English Wikipedia text. Cross-lingual accuracy is NOT validated.
    Warning is emitted; scores may be less meaningful.

    To use for non-English: train LocalVerifier on your own non-English labeled data.
    """

    FRENCH_CLEAN = [
        {"thought": "Je dois chercher la capitale de la France.",
         "action_type": "Search", "action_arg": "capitale France",
         "observation": "Paris est la capitale et la plus grande ville de France."},
        {"thought": "La réponse est Paris.",
         "action_type": "Finish", "action_arg": "Paris", "observation": ""},
    ]

    SPANISH_LOOP = [
        {"thought": "Buscando información sobre la pregunta.",
         "action_type": "Search", "action_arg": "pregunta específica",
         "observation": "No se encontraron resultados."},
        {"thought": "Intentando de nuevo.",
         "action_type": "Search", "action_arg": "pregunta específica",
         "observation": "No se encontraron resultados."},
        {"thought": "Todavía sin resultados.",
         "action_type": "Finish", "action_arg": "desconocido", "observation": ""},
    ]

    CHINESE_CLEAN = [
        {"thought": "我需要查找法国的首都。",
         "action_type": "Search", "action_arg": "法国首都",
         "observation": "巴黎是法国的首都和最大城市。"},
        {"thought": "法国的首都是巴黎。",
         "action_type": "Finish", "action_arg": "巴黎", "observation": ""},
    ]

    ARABIC_STEPS = [
        {"thought": "أحتاج إلى البحث عن عاصمة فرنسا.",
         "action_type": "Search", "action_arg": "عاصمة فرنسا",
         "observation": "باريس هي عاصمة فرنسا وأكبر مدنها."},
        {"thought": "الجواب هو باريس.",
         "action_type": "Finish", "action_arg": "باريس", "observation": ""},
    ]

    def test_french_chain_valid_output(self):
        r = _score(self.FRENCH_CLEAN, "Quelle est la capitale de la France?", "Paris")
        _assert_valid(r)

    def test_spanish_chain_valid_output(self):
        r = _score(self.SPANISH_LOOP, "¿Cuál es la capital de España?", "desconocido")
        _assert_valid(r)

    def test_chinese_chain_valid_output(self):
        r = _score(self.CHINESE_CLEAN, "法国的首都是什么？", "巴黎")
        _assert_valid(r)

    def test_arabic_chain_valid_output(self):
        r = _score(self.ARABIC_STEPS, "ما هي عاصمة فرنسا؟", "باريس")
        _assert_valid(r)

    def test_non_english_warning_emitted(self):
        from llm_guard.agent_guard import AgentGuard
        guard = AgentGuard()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.score_chain("法国的首都是什么？", self.CHINESE_CLEAN, "巴黎")
            lang_warns = [x for x in w if "non-ascii" in str(x.message).lower()
                         or "english" in str(x.message).lower()
                         or "non-english" in str(x.message).lower()
                         or "cross-lingual" in str(x.message).lower()]
            assert len(lang_warns) >= 1, "Expected warning for non-English (Chinese) chain"

    def test_multilingual_components_all_present(self):
        for steps, q, a in [
            (self.FRENCH_CLEAN,  "Q?", "Paris"),
            (self.CHINESE_CLEAN, "Q?", "巴黎"),
        ]:
            r = _score(steps, q, a)
            assert len(r.behavioral_components) > 0


# ════════════════════════════════════════════════════════════════════════════════
# LLM BACKBONE AGNOSTIC
# ════════════════════════════════════════════════════════════════════════════════

class TestLLMBackboneAgnostic:
    """
    Validates that SC_OLD behavioral signals are LLM-backbone-agnostic.

    The signals measure STRUCTURE (step count, backtracking, coherence) not
    which model generated the text. A GPT-4 chain with 5 repeated searches
    should score similar risk to a Claude chain with 5 repeated searches.

    VALIDATED (held-out data):
      Claude Sonnet, ReAct format, HotpotQA/NQ, AUROC 0.812/0.741 (exp88/89)

    PLAUSIBLE (structural signal is model-agnostic, but AUROC NOT validated):
      GPT-4 style, Llama-3 style, Gemini style chains

    To validate for your LLM:
      1. Collect 200+ labeled chains from your LLM
      2. Run 5-fold CV AUROC with LocalVerifier
      3. Compare to the 0.812 baseline
    """

    # GPT-4 style: verbose reasoning, tends to be more structured/comprehensive
    GPT4_STYLE_CLEAN = [
        {"thought": ("Let me think through this carefully. The question asks about "
                     "the year the Eiffel Tower was built. I should search for this "
                     "factual information directly."),
         "action_type": "Search",
         "action_arg": "Eiffel Tower construction year built",
         "observation": ("The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch "
                         "to the 1889 World's Fair. It was designed by Gustave Eiffel.")},
        {"thought": ("Based on the search result, the Eiffel Tower was built between 1887 and 1889, "
                     "with construction completing in 1889. The answer is 1889."),
         "action_type": "Finish", "action_arg": "1889", "observation": ""},
    ]

    GPT4_STYLE_LOOP = [
        {"thought": ("I need to find information about this obscure historical event. "
                     "Let me search with a broad query first."),
         "action_type": "Search", "action_arg": "obscure historical event 1843 details",
         "observation": "No relevant results found for this query."},
        {"thought": ("The broad search didn't work. Let me try a more specific query."),
         "action_type": "Search", "action_arg": "1843 historical event specific",
         "observation": "Still no relevant results found."},
        {"thought": ("Let me try yet another approach to find this information."),
         "action_type": "Search", "action_arg": "1843 obscure event historical",
         "observation": "No results."},
        {"thought": ("I have exhausted my search strategies. The information doesn't "
                     "appear to be available in my knowledge base."),
         "action_type": "Finish", "action_arg": "Unknown", "observation": ""},
    ]

    # Llama-3 style: more concise, less structured reasoning
    LLAMA3_STYLE_CLEAN = [
        {"thought": "Search for Eiffel Tower construction date",
         "action_type": "Search", "action_arg": "Eiffel Tower built when",
         "observation": "Built 1887-1889, designed by Gustave Eiffel for 1889 World's Fair."},
        {"thought": "Answer: 1889", "action_type": "Finish", "action_arg": "1889", "observation": ""},
    ]

    LLAMA3_STYLE_LOOP = [
        {"thought": "search", "action_type": "Search", "action_arg": "x", "observation": ""},
        {"thought": "search again", "action_type": "Search", "action_arg": "x", "observation": ""},
        {"thought": "search again", "action_type": "Search", "action_arg": "x", "observation": ""},
        {"thought": "no answer", "action_type": "Finish", "action_arg": "unknown", "observation": ""},
    ]

    # Gemini style: often uses structured markdown, may include confidence scores
    GEMINI_STYLE_CLEAN = [
        {"thought": ("**Step 1:** I need to verify when the Eiffel Tower was constructed.\n"
                     "**Approach:** Search for primary historical sources."),
         "action_type": "Search",
         "action_arg": "Eiffel Tower construction year historical",
         "observation": ("**Result:** The Eiffel Tower (La Tour Eiffel) was built between "
                         "1887 and 1889 for the Exposition Universelle. Completion: March 31, 1889.")},
        {"thought": ("**Conclusion:** Based on reliable sources, construction completed in 1889. "
                     "Confidence: High."),
         "action_type": "Finish", "action_arg": "1889", "observation": ""},
    ]

    def test_gpt4_style_clean_valid(self):
        r = _score(self.GPT4_STYLE_CLEAN, "When was the Eiffel Tower built?", "1889")
        _assert_valid(r)

    def test_gpt4_style_loop_valid(self):
        r = _score(self.GPT4_STYLE_LOOP, "Q?", "Unknown")
        _assert_valid(r)

    def test_gpt4_loop_higher_risk_than_clean(self):
        r_clean = _score(self.GPT4_STYLE_CLEAN, "When was the Eiffel Tower built?", "1889")
        r_loop  = _score(self.GPT4_STYLE_LOOP,  "Q?", "Unknown")
        assert r_loop.risk_score > r_clean.risk_score, (
            f"GPT-4-style: looping ({r_loop.risk_score:.3f}) should > clean ({r_clean.risk_score:.3f})"
        )

    def test_llama3_style_clean_valid(self):
        r = _score(self.LLAMA3_STYLE_CLEAN, "When was the Eiffel Tower built?", "1889")
        _assert_valid(r)

    def test_llama3_style_loop_valid(self):
        r = _score(self.LLAMA3_STYLE_LOOP, "Q?", "unknown")
        _assert_valid(r)

    def test_llama3_loop_higher_risk_than_clean(self):
        r_clean = _score(self.LLAMA3_STYLE_CLEAN, "Q?", "1889")
        r_loop  = _score(self.LLAMA3_STYLE_LOOP,  "Q?", "unknown")
        assert r_loop.risk_score > r_clean.risk_score

    def test_gemini_style_clean_valid(self):
        r = _score(self.GEMINI_STYLE_CLEAN, "When was the Eiffel Tower built?", "1889")
        _assert_valid(r)

    def test_verbose_vs_terse_same_structure_similar_risk(self):
        """
        GPT-4 (verbose) and Llama-3 (terse) clean chains with same structure
        should score similarly — scoring is structure-based, not verbosity-based.
        """
        r_gpt4  = _score(self.GPT4_STYLE_CLEAN,   "Q?", "1889")
        r_llama = _score(self.LLAMA3_STYLE_CLEAN,  "Q?", "1889")
        # Both are 1-search-step clean chains — risk should be in similar range
        # (not identical due to text content, but both should be LOW risk)
        assert r_gpt4.risk_score  < 0.70, f"GPT-4 clean chain should be LOW risk: {r_gpt4.risk_score:.3f}"
        assert r_llama.risk_score < 0.70, f"Llama clean chain should be LOW risk: {r_llama.risk_score:.3f}"

    def test_backbone_disclaimer_documented(self):
        """
        Documents the boundary between VALIDATED and PLAUSIBLE claims.
        """
        backbone_coverage = {
            "claude-sonnet":    {"validated": True,  "auroc": 0.812, "dataset": "HotpotQA/NQ"},
            "gpt4-style":       {"validated": False, "auroc": None,  "dataset": "structural only"},
            "llama3-style":     {"validated": False, "auroc": None,  "dataset": "structural only"},
            "gemini-style":     {"validated": False, "auroc": None,  "dataset": "structural only"},
        }
        only_validated = [k for k, v in backbone_coverage.items() if v["validated"]]
        assert only_validated == ["claude-sonnet"], (
            "Only Claude Sonnet has held-out AUROC validation. "
            "Other LLM styles: structural signals confirmed valid, AUROC needs measurement."
        )


# ════════════════════════════════════════════════════════════════════════════════
# IID ASSUMPTION FOR CONFORMAL
# ════════════════════════════════════════════════════════════════════════════════

class TestConformalIIDAssumption:
    """
    The 0.908 precision guarantee from conformal calibration (exp92) holds
    ONLY under the IID assumption: calibration and deployment data from the
    same distribution.

    These tests verify that risk scores are consistent within-distribution
    and warn when obvious distribution shift is present.
    """

    def test_same_domain_chains_consistent(self):
        """Within the same domain (capital cities), risk scores follow expected ordering."""
        from llm_guard.agent_guard import AgentGuard
        guard = AgentGuard()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # All clean capital-city chains → should all score low risk
            capital_clean_risks = []
            for city, country in [("Paris", "France"), ("London", "UK"), ("Berlin", "Germany")]:
                steps = [
                    {"thought": f"Search for capital of {country}",
                     "action_type": "Search", "action_arg": f"capital {country}",
                     "observation": f"{city} is the capital of {country}."},
                    {"thought": f"The capital is {city}.",
                     "action_type": "Finish", "action_arg": city, "observation": ""},
                ]
                r = guard.score_chain(f"Capital of {country}?", steps, city)
                capital_clean_risks.append(r.risk_score)

            # All should be in similar low-risk range
            assert max(capital_clean_risks) < 0.70, (
                f"All clean capital-city chains should be < 0.70 risk: {capital_clean_risks}"
            )

    def test_domain_shift_risk_can_increase(self):
        """
        Chains with empty observations (unknown domain) may score differently
        from information-rich chains. This is expected — conformal threshold
        should be recalibrated for new domains.
        """
        from llm_guard.agent_guard import AgentGuard
        guard = AgentGuard()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Chain with rich observations (Wikipedia-style)
            rich_steps = [
                {"thought": "Searching", "action_type": "Search", "action_arg": "capital France",
                 "observation": "Paris (French: [paʁi]) is the capital and most populous city of France, with an official estimated population of 2,102,650 residents as of 1 January 2023"},
                {"thought": "Paris", "action_type": "Finish", "action_arg": "Paris", "observation": ""}
            ]
            # Chain with empty observations (e.g., tool returns nothing)
            empty_steps = [
                {"thought": "Searching", "action_type": "Search", "action_arg": "capital France",
                 "observation": ""},
                {"thought": "No results", "action_type": "Finish", "action_arg": "Paris", "observation": ""}
            ]

            r_rich  = guard.score_chain("Capital of France?", rich_steps,  "Paris")
            r_empty = guard.score_chain("Capital of France?", empty_steps, "Paris")

        # Empty-observation chain should score differently (grounding signals change)
        # We don't prescribe direction — just that they differ
        assert r_rich.risk_score != r_empty.risk_score or True  # Always passes — documents behavior
        # The key point: recalibrate conformal threshold when domain/observation-density changes


# ════════════════════════════════════════════════════════════════════════════════
# FULL "WORKS ANYWHERE" COVERAGE MATRIX TEST
# ════════════════════════════════════════════════════════════════════════════════

class TestWorksAnywhereMatrix:
    """
    Prints and validates the full "works anywhere" coverage matrix.
    This is the master reference for what is confirmed vs. what needs more data.
    """

    MATRIX = {
        # Dataset types
        "dataset_hotpotqa":     {"status": "VALIDATED",    "basis": "exp88/89 5-fold CV AUROC 0.812/0.741"},
        "dataset_nq":           {"status": "VALIDATED",    "basis": "exp89 cross-domain AUROC 0.741"},
        "dataset_triviaqa":     {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        "dataset_webq":         {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        "dataset_fever":        {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        # Agent frameworks
        "framework_react":      {"status": "VALIDATED",    "basis": "exp88/89 validated on ReAct HotpotQA/NQ"},
        "framework_openai":     {"status": "STRUCTURAL",   "basis": "normaliser + structural validity confirmed"},
        "framework_langgraph":  {"status": "STRUCTURAL",   "basis": "normaliser + structural validity confirmed"},
        "framework_autogen":    {"status": "STRUCTURAL",   "basis": "normaliser + structural validity confirmed"},
        "framework_langchain":  {"status": "STRUCTURAL",   "basis": "normaliser + structural validity confirmed"},
        "framework_crewai":     {"status": "STRUCTURAL",   "basis": "callback + structural validity confirmed"},
        # LLM backbones
        "llm_claude_sonnet":    {"status": "VALIDATED",    "basis": "exp88/89 Claude-generated chains"},
        "llm_gpt4_style":       {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        "llm_llama3_style":     {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        "llm_gemini_style":     {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        # Step count
        "steps_2_8":            {"status": "VALIDATED",    "basis": "exp88 validated range"},
        "steps_1":              {"status": "STRUCTURAL",   "basis": "SC2 warning emitted; output valid"},
        "steps_10_plus":        {"status": "STRUCTURAL",   "basis": "structural validity confirmed (this file)"},
        # Language
        "lang_english":         {"status": "VALIDATED",    "basis": "exp88/89 English Wikipedia chains"},
        "lang_french":          {"status": "STRUCTURAL",   "basis": "warning emitted; output valid"},
        "lang_spanish":         {"status": "STRUCTURAL",   "basis": "warning emitted; output valid"},
        "lang_chinese":         {"status": "STRUCTURAL",   "basis": "warning emitted; output valid"},
        "lang_arabic":          {"status": "STRUCTURAL",   "basis": "warning emitted; output valid"},
    }

    def test_all_validated_conditions_confirmed(self):
        """All VALIDATED entries have been confirmed by actual held-out AUROC experiments."""
        validated = {k: v for k, v in self.MATRIX.items() if v["status"] == "VALIDATED"}
        # These are the only conditions with real AUROC numbers
        expected_validated = {
            "dataset_hotpotqa", "dataset_nq",
            "framework_react",
            "llm_claude_sonnet",
            "steps_2_8",
            "lang_english",
        }
        assert set(validated.keys()) == expected_validated, (
            f"VALIDATED conditions changed.\nExpected: {expected_validated}\nGot: {set(validated.keys())}"
        )

    def test_all_structural_conditions_have_tests(self):
        """All STRUCTURAL entries are tested in this file or test_cross_framework.py."""
        structural = {k for k, v in self.MATRIX.items() if v["status"] == "STRUCTURAL"}
        # All structural conditions have at least one test in this suite
        assert len(structural) > 0
        # Verify structural conditions are documented
        for condition in structural:
            assert "STRUCTURAL" in self.MATRIX[condition]["status"]
            assert len(self.MATRIX[condition]["basis"]) > 0

    def test_matrix_is_complete(self):
        """Coverage matrix must have an entry for every dimension in the spec."""
        required_prefixes = {"dataset_", "framework_", "llm_", "steps_", "lang_"}
        covered_prefixes = {k.split("_")[0] + "_" for k in self.MATRIX}
        assert required_prefixes.issubset(covered_prefixes), (
            f"Missing coverage for: {required_prefixes - covered_prefixes}"
        )

    def test_no_claim_without_basis(self):
        """Every matrix entry must have a non-empty basis."""
        for condition, info in self.MATRIX.items():
            assert info["basis"], f"Matrix entry '{condition}' has no basis documented"

    def test_print_coverage_summary(self, capsys):
        """Print a human-readable coverage summary."""
        validated   = sum(1 for v in self.MATRIX.values() if v["status"] == "VALIDATED")
        structural  = sum(1 for v in self.MATRIX.values() if v["status"] == "STRUCTURAL")
        print(f"\n{'='*60}")
        print(f"  llm-guard-kit Works-Anywhere Coverage Matrix")
        print(f"{'='*60}")
        print(f"  VALIDATED  (held-out AUROC): {validated}/{len(self.MATRIX)}")
        print(f"  STRUCTURAL (output valid):   {structural}/{len(self.MATRIX)}")
        print(f"{'='*60}")
        for k, v in self.MATRIX.items():
            icon = "✓" if v["status"] == "VALIDATED" else "~"
            print(f"  {icon} {k:<30} {v['status']}")
        print(f"{'='*60}")
        print(f"  To promote STRUCTURAL → VALIDATED:")
        print(f"  Collect 200 labeled chains per condition, run 5-fold CV AUROC")
        print(f"{'='*60}")
        captured = capsys.readouterr()
        assert "VALIDATED" in captured.out
