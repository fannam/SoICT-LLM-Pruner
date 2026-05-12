from __future__ import annotations

from carve_lm._distillation import DistilModel as SharedDistilModel
from carve_lm._distillation import OTConfig as SharedOTConfig
from carve_lm.llm.distillation import DistilModel as LLMDistilModel
from carve_lm.llm.distillation import OTConfig as LLMOTConfig
from carve_lm.vlm.distillation import DistilModel as VLMDistilModel
from carve_lm.vlm.distillation import OTConfig as VLMOTConfig


def test_llm_and_vlm_distillation_facades_share_ot_and_wrapper_types():
    assert LLMDistilModel is SharedDistilModel
    assert VLMDistilModel is SharedDistilModel
    assert LLMOTConfig is SharedOTConfig
    assert VLMOTConfig is SharedOTConfig
