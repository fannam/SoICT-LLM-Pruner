from __future__ import annotations

import torch
from tests.vlm.fixtures.synthetic_vlm import make_synthetic_vlm

from carve_lm.vlm.distillation import HybridDistiller, LogitsDistiller


def make_multimodal_batch(offset: int, *, vocab_size: int = 16, padded: bool = False) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [
                (offset + 1) % vocab_size,
                (offset + 2) % vocab_size,
                (offset + 3) % vocab_size,
                (offset + 4) % vocab_size,
            ],
            [
                (offset + 2) % vocab_size,
                (offset + 3) % vocab_size,
                (offset + 4) % vocab_size,
                (offset + 5) % vocab_size,
            ],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    if padded:
        attention_mask[:, -1] = 0
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    pixel_values = torch.tensor(
        [
            [offset + 0.1, offset + 0.2, offset + 0.3, offset + 0.4],
            [offset + 1.1, offset + 1.2, offset + 1.3, offset + 1.4],
        ],
        dtype=torch.float32,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


def test_logits_distiller_preserves_multimodal_batch_keys():
    torch.manual_seed(0)
    teacher = make_synthetic_vlm(num_hidden_layers=1)
    student = make_synthetic_vlm(num_hidden_layers=1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    distiller = LogitsDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=optimizer,
    )

    val_history = distiller.distill(
        train_loader=[make_multimodal_batch(0), make_multimodal_batch(1, padded=True)],
        val_loader=[make_multimodal_batch(7, padded=True)],
        device_teacher="cpu",
        device_student="cpu",
        epochs=1,
        grad_accumulation_steps=1,
    )

    assert len(val_history) == 1
    assert distiller.history["train_loss"]


def test_hybrid_distiller_preserves_multimodal_batch_keys():
    torch.manual_seed(0)
    teacher = make_synthetic_vlm(hidden_size=8, num_hidden_layers=2)
    student = make_synthetic_vlm(hidden_size=4, num_hidden_layers=1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)

    distiller = HybridDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=optimizer,
    )

    val_history = distiller.distill(
        train_loader=[make_multimodal_batch(0), make_multimodal_batch(1, padded=True)],
        val_loader=[make_multimodal_batch(7, padded=True)],
        device_teacher="cpu",
        device_student="cpu",
        epochs=1,
        grad_accumulation_steps=1,
        block_layers_to_prune=[1],
    )

    assert distiller.teacher_kept_layers == [0]
    assert len(val_history) == 1
    assert distiller.history["train_loss"]
