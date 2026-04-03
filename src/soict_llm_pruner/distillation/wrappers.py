from __future__ import annotations

import torch.nn as nn


class DistilModel(nn.Module):
    def __init__(self, student, student_dim, teacher_dim, teacher_kept_layers):
        super().__init__()
        self.student = student
        self.teacher_kept_layers = teacher_kept_layers

        if student_dim == teacher_dim:
            self.projectors = nn.ModuleList(
                [nn.Identity() for _ in teacher_kept_layers]
            )
        else:
            self.projectors = nn.ModuleList(
                [
                    nn.Linear(student_dim, teacher_dim)
                    for _ in teacher_kept_layers
                ]
            )

    def forward(self, **kwargs):
        return self.student(**kwargs, output_hidden_states=True)
