"""Tests for StepExtractor ABC and LLMReActExtractor."""
import sys, pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STEPS = [
    {"thought": "Search for Einstein birthday.", "action_type": "Search",
     "action_arg": "Einstein birthday",
     "observation": "Albert Einstein was born on March 14, 1879."},
    {"thought": "Born 1879.", "action_type": "Finish", "action_arg": "1879", "observation": ""},
]


def test_abc_cannot_instantiate():
    from llm_guard.step_extractor import StepExtractor
    with pytest.raises(TypeError):
        StepExtractor()


def test_custom_extractor_via_abc():
    from llm_guard.step_extractor import StepExtractor

    class MyExtractor(StepExtractor):
        @property
        def feature_names(self):
            return ["n_words", "is_question"]

        def extract(self, step):
            text = step.get("thought", "")
            return {
                "n_words":     min(len(text.split()) / 50.0, 1.0),
                "is_question": float("?" in text),
            }

    ext   = MyExtractor()
    feats = ext.extract({"thought": "What is going on here?"})
    assert "n_words" in feats
    assert feats["is_question"] == 1.0


def test_aggregate_produces_vector():
    from llm_guard.step_extractor import LLMReActExtractor
    ext = LLMReActExtractor()
    vec = ext.aggregate(STEPS, final_answer="1879")
    assert vec.shape == (len(ext.feature_names),)


def test_llmreact_feature_names():
    from llm_guard.step_extractor import LLMReActExtractor
    ext   = LLMReActExtractor()
    names = ext.feature_names
    for expected in ("sc2_step_count", "sc4_uncertainty_density", "sc5_backtrack_rate",
                     "retrieval_conf", "semantic_gap"):
        assert expected in names, f"Missing {expected} in feature_names"


def test_extract_returns_dict_float_values():
    from llm_guard.step_extractor import LLMReActExtractor
    ext  = LLMReActExtractor()
    step = {"thought": "Searching for facts.", "action_type": "Search",
            "action_arg": "facts", "observation": "Some facts here."}
    feats = ext.extract(step)
    assert isinstance(feats, dict)
    for k, v in feats.items():
        assert isinstance(v, float), f"Feature {k} is not float: {type(v)}"
