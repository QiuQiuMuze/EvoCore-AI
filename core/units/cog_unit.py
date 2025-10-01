from .base_unit import CogUnitBase
from .evaluation import EvaluationMixin
from .learning import LearningMixin
from .memory import MemoryMixin
from .mortality import MortalityMixin
from .output import OutputMixin
from .reproduction import ReproductionMixin


class CogUnit(
    LearningMixin,
    OutputMixin,
    ReproductionMixin,
    EvaluationMixin,
    MemoryMixin,
    MortalityMixin,
    CogUnitBase,
):
    """组合所有行为的完整细胞单元"""


class SensorUnit(CogUnit):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("role", "sensor")
        super().__init__(*args, **kwargs)


class ProcessorUnit(CogUnit):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("role", "processor")
        super().__init__(*args, **kwargs)


class EmitterUnit(CogUnit):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("role", "emitter")
        super().__init__(*args, **kwargs)
