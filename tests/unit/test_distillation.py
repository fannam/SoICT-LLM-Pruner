from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from soict_llm_pruner.distillation import HybridDistiller, LogitsDistiller


@dataclass
class TinyConfig:
    hidden_size: int = 8
    num_hidden_layers: int = 2
    vocab_size: int = 17
    block_layers_to_prune: list[int] | None = None


class TinyCausalLM(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.saved = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
        hidden_states = self.embedding(input_ids)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = torch.tanh(layer(hidden_states))
            all_hidden_states.append(hidden_states)

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[..., 1:].reshape(-1)
            valid_mask = shift_labels.ne(-100)
            if attention_mask is not None:
                valid_mask = valid_mask & attention_mask[..., 1:].reshape(-1).bool()
            if torch.any(valid_mask):
                loss = F.cross_entropy(shift_logits[valid_mask], shift_labels[valid_mask])
            else:
                loss = shift_logits.sum() * 0

        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            loss=loss,
        )

    def save_pretrained(self, output_dir, **kwargs):
        self.saved = (Path(output_dir), kwargs)


class CountingScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class DummyAccelerator:
    def __init__(self):
        self.is_main_process = True
        self.is_local_main_process = True
        self.num_processes = 1
        self.logged = []
        self.saved = []
        self.waited = False
        self.ended = False

    def prepare(self, *args):
        return args

    def init_trackers(self, *args, **kwargs):
        self.trackers = (args, kwargs)

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def print(self, *args, **kwargs):
        return None

    def wait_for_everyone(self):
        self.waited = True

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        return model.state_dict()

    def save(self, obj, path):
        self.saved.append((obj, path))

    def end_training(self):
        self.ended = True

    def accumulate(self, model):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

    def gather(self, tensor):
        return tensor


class DummyTokenizer:
    def __init__(self):
        self.saved_path = None

    def save_pretrained(self, output_dir):
        self.saved_path = Path(output_dir)


def make_batch(offset: int, *, vocab_size: int, padded: bool = False) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [(offset + 1) % vocab_size, (offset + 2) % vocab_size, (offset + 3) % vocab_size, (offset + 4) % vocab_size],
            [(offset + 2) % vocab_size, (offset + 3) % vocab_size, (offset + 4) % vocab_size, (offset + 5) % vocab_size],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    if padded:
        attention_mask[:, -1] = 0
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def attach_step_counter(optimizer):
    original_step = optimizer.step
    state = {"calls": 0}

    def counted_step(*args, **kwargs):
        state["calls"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    return state


def test_logits_loss_ignores_masked_positions():
    distiller = LogitsDistiller(teacher_model=None, student_model=None)
    student_logits = torch.randn(1, 3, 7)
    teacher_logits = torch.randn(1, 3, 7)
    labels_a = torch.tensor([[1, 2, 3]])
    labels_b = torch.tensor([[1, 2, 6]])
    loss_mask = torch.tensor([[True, True, False]])

    loss_a = distiller.logits_loss(student_logits, teacher_logits, labels_a, loss_mask)
    loss_b = distiller.logits_loss(student_logits, teacher_logits, labels_b, loss_mask)

    assert torch.isclose(loss_a, loss_b)


def test_logits_distiller_flushes_remainder_gradients():
    torch.manual_seed(0)
    teacher = TinyCausalLM(TinyConfig(hidden_size=8, num_hidden_layers=2, vocab_size=23))
    student = TinyCausalLM(TinyConfig(hidden_size=8, num_hidden_layers=2, vocab_size=23))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = CountingScheduler(optimizer)
    step_state = attach_step_counter(optimizer)
    train_loader = [make_batch(idx, vocab_size=23, padded=(idx % 2 == 0)) for idx in range(3)]
    val_loader = [make_batch(10, vocab_size=23, padded=True)]

    distiller = LogitsDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    val_history = distiller.distill(
        train_loader=train_loader,
        val_loader=val_loader,
        device_teacher="cpu",
        device_student="cpu",
        epochs=1,
        grad_accumulation_steps=2,
    )

    assert step_state["calls"] == 2
    assert scheduler.step_calls == 2
    assert len(val_history) == 1
    assert distiller.history["train_loss"]


def test_hybrid_distiller_adds_projectors_to_optimizer_and_flushes_remainder():
    torch.manual_seed(0)
    teacher = TinyCausalLM(TinyConfig(hidden_size=8, num_hidden_layers=3, vocab_size=29))
    student = TinyCausalLM(TinyConfig(hidden_size=4, num_hidden_layers=2, vocab_size=29))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    scheduler = CountingScheduler(optimizer)
    step_state = attach_step_counter(optimizer)
    train_loader = [make_batch(idx, vocab_size=29, padded=(idx == 1)) for idx in range(3)]
    val_loader = [make_batch(7, vocab_size=29, padded=True)]

    distiller = HybridDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    val_history = distiller.distill(
        train_loader=train_loader,
        val_loader=val_loader,
        device_teacher="cpu",
        device_student="cpu",
        epochs=1,
        grad_accumulation_steps=2,
        block_layers_to_prune=[1],
    )

    projector_ids = {id(parameter) for parameter in distiller.distill_model.projectors.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert distiller.teacher_kept_layers == [0, 2]
    assert len(optimizer.param_groups) == 2
    assert projector_ids <= optimizer_ids
    assert step_state["calls"] == 2
    assert scheduler.step_calls == 2
    assert len(val_history) == 1


def test_teacher_correction_save_model_persists_tokenizer(tmp_path):
    pytest.importorskip("accelerate")

    from soict_llm_pruner.distillation import TeacherCorrection

    model = TinyCausalLM(TinyConfig(hidden_size=8, num_hidden_layers=2, vocab_size=19))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler(optimizer)
    accelerator = DummyAccelerator()
    tokenizer = DummyTokenizer()
    batch = make_batch(0, vocab_size=19, padded=True)

    trainer = TeacherCorrection(
        model=model,
        train_loader=[batch],
        val_loader=[batch],
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        tokenizer=tokenizer,
        config={"num_epochs": 1, "gradient_accumulation_steps": 1},
    )

    trainer.save_model(tmp_path)

    saved_path, save_kwargs = model.saved
    assert saved_path == tmp_path
    assert "save_function" in save_kwargs
    assert "state_dict" in save_kwargs
    assert tokenizer.saved_path == tmp_path
    assert accelerator.waited is True
