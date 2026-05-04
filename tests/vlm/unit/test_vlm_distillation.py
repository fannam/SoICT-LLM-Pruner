from __future__ import annotations

import torch
from tests.vlm.fixtures.synthetic_vlm import make_synthetic_vlm

from carve_lm.vlm.distillation import HybridDistiller, LogitsDistiller, create_distillation_dataloader


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        texts,
        *,
        padding=True,
        truncation=True,
        max_length=None,
        return_tensors=None,
    ):
        del padding, truncation
        rows = []
        for text in texts:
            ids = [ord(char) % 11 + 3 for char in text]
            if max_length is not None:
                ids = ids[:max_length]
            rows.append(torch.tensor(ids, dtype=torch.long))
        width = max(row.numel() for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, row in enumerate(rows):
            input_ids[row_idx, : row.numel()] = row
            attention_mask[row_idx, : row.numel()] = 1
        if return_tensors != "pt":
            raise ValueError("DummyTokenizer only supports return_tensors='pt'.")
        return {"input_ids": input_ids, "attention_mask": attention_mask}


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


def test_vlm_distillation_dataloader_preserves_pixel_values():
    loader = create_distillation_dataloader(
        [
            {
                "text": "abcd",
                "pixel_values": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            },
            {
                "text": "ef",
                "pixel_values": torch.tensor([1.1, 1.2, 1.3, 1.4]),
            },
        ],
        tokenizer=DummyTokenizer(),
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 4)
    assert batch["pixel_values"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["labels"][1, -1].item() == -100


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
