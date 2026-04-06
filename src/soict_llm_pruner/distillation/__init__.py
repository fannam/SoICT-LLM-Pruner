from __future__ import annotations

__all__ = [
    "DistilModel",
    "HybridDistiller",
    "LogitsDistiller",
    "TeacherCorrection",
]


def __getattr__(name: str):
    if name == "DistilModel":
        from .wrappers import DistilModel

        return DistilModel
    if name == "HybridDistiller":
        from .hybrid import HybridDistiller

        return HybridDistiller
    if name == "LogitsDistiller":
        from .logits import LogitsDistiller

        return LogitsDistiller
    if name == "TeacherCorrection":
        try:
            from .teacher_correction import TeacherCorrection
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TeacherCorrection requires the optional training dependencies. "
                "Install the `train` extra to use it."
            ) from exc

        return TeacherCorrection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
