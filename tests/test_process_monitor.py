import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STEPS = [
    {"thought": "Search for Einstein birthday.", "action_type": "Search",
     "action_arg": "Einstein birthday",
     "observation": "Albert Einstein was born on March 14, 1879."},
    {"thought": "Born 1879.", "action_type": "Finish", "action_arg": "1879", "observation": ""},
]

def test_monitor_result_schema():
    from llm_guard.process_monitor import ProcessReliabilityMonitor, MonitorResult
    from llm_guard.step_extractor import LLMReActExtractor
    monitor = ProcessReliabilityMonitor(extractor=LLMReActExtractor())
    result  = monitor.score(steps=STEPS, output="1879")
    assert isinstance(result, MonitorResult)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.confidence_tier in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(result.needs_alert, bool)

def test_monitor_custom_extractor():
    from llm_guard.process_monitor import ProcessReliabilityMonitor, MonitorResult
    from llm_guard.step_extractor import StepExtractor

    class SimpleExtractor(StepExtractor):
        @property
        def feature_names(self):
            return ["is_empty_output"]
        def extract(self, step):
            return {"is_empty_output": float(not step.get("output", ""))}

    monitor = ProcessReliabilityMonitor(extractor=SimpleExtractor())
    result  = monitor.score(steps=STEPS, output="1879")
    assert isinstance(result, MonitorResult)

def test_monitor_domain_registration():
    from llm_guard.process_monitor import ProcessReliabilityMonitor
    from llm_guard.step_extractor import LLMReActExtractor
    monitor = ProcessReliabilityMonitor.for_domain("llm_react")
    assert monitor is not None
    assert isinstance(monitor.extractor, LLMReActExtractor)

def test_monitor_score_with_judge():
    from llm_guard.process_monitor import ProcessReliabilityMonitor
    from llm_guard.step_extractor import LLMReActExtractor
    def mock_judge(steps, output):
        return 0.3
    monitor = ProcessReliabilityMonitor(
        extractor=LLMReActExtractor(),
        judge_fn=mock_judge,
        judge_weight=0.5,
    )
    result = monitor.score(steps=STEPS, output="1879")
    assert 0.0 <= result.risk_score <= 1.0

def test_monitor_exported_from_package():
    from llm_guard import ProcessReliabilityMonitor, StepExtractor, LLMReActExtractor
    assert ProcessReliabilityMonitor is not None
    assert StepExtractor is not None
    assert LLMReActExtractor is not None
