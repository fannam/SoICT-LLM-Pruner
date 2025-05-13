import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import DistilModel

class HybridDistiller:
    def __init__(self, teacher_model, student_model, tokenizer, optimizer, scheduler, use_wandb=False, wandb_key=None, project_name=None, run_name=None):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        # wandb optional
        self.use_wandb = use_wandb and wandb_key is not None
        self.wandb_key = wandb_key
        self.project_name = project_name
        self.run_name = run_name

    def logits_loss(self, student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        log_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        forward_kl = F.kl_div(log_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)
        return alpha * ce_loss + (1 - alpha) * forward_kl

    def feature_loss_selected_layers(self, student_feats, teacher_feats, projectors, mask):
        losses = []
        for proj, s_feat, t_feat in zip(projectors, student_feats, teacher_feats):
            s_feat = s_feat.float()
            p = proj(s_feat)
            p = p[mask]
            t = t_feat.float()[mask]
            losses.append(F.mse_loss(p, t, reduction='mean'))
        return torch.stack(losses).mean()

    def distill(
        self,
        train_loader,
        val_loader,
        device_teacher,
        device_student,
        epochs=1,
        grad_accumulation_steps=8,
        alpha=0.1,
        gamma=0.1,
        temperature=2.0,
        block_layers_to_prune=None
    ):
        self.teacher_model.eval()

        total_layers = self.teacher_model.config.num_hidden_layers
        prune = block_layers_to_prune or getattr(self.student_model.config, 'block_layers_to_prune', None)
        teacher_kept = [i for i in range(total_layers) if not prune or i not in prune]
        if self.student_model.config.num_hidden_layers != len(teacher_kept):
            raise ValueError(f"Student has {self.student_model.config.num_hidden_layers} layers, expected {len(teacher_kept)}.")

        student_dim = self.student_model.config.hidden_size
        teacher_dim = self.teacher_model.config.hidden_size
        wrapper = DistilModel(self.student_model, student_dim, teacher_dim, teacher_kept).to(device_student)
        self.teacher_model.to(device_teacher)
        wrapper.train()

        if self.use_wandb:
            import wandb
            wandb.login(key=self.wandb_key)
            wandb.init(project=self.project_name, name=self.run_name, reinit=True)

        val_loss_history = []
        step = 0
        for epoch in range(epochs):

            for batch in train_loader:
                ids = batch['input_ids'].to(device_student)
                mask = batch['attention_mask'].to(device_student)
                labels = batch['labels'][:,1:].to(device_student).reshape(-1)
                with torch.no_grad():
                    t_out = self.teacher_model(input_ids=batch['input_ids'].to(device_teacher),
                                               attention_mask=batch['attention_mask'].to(device_teacher),
                                               output_hidden_states=True)
                t_logits = t_out.logits[:,:-1,:].reshape(-1, t_out.logits.size(-1)).to(device_student)
                s_out = wrapper(input_ids=ids, attention_mask=mask)
                s_logits = s_out.logits[:,:-1,:].reshape(-1, s_out.logits.size(-1))
                kl_ce = self.logits_loss(s_logits, t_logits, labels, temperature, alpha)
                student_feats = [s_out.hidden_states[i+1][:,:-1,:] for i in range(len(teacher_kept))]
                teacher_feats = [t_out.hidden_states[i+1][:,:-1,:].to(device_student) for i in teacher_kept]
                pad_mask = mask[:,1:].bool()
                f_loss = self.feature_loss_selected_layers(student_feats, teacher_feats, wrapper.projectors, pad_mask)
                loss = (kl_ce + gamma*f_loss) / grad_accumulation_steps
                loss.backward()
                if (step+1) % grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if self.use_wandb:
                        wandb.log({'train_loss': loss.item()*grad_accumulation_steps,
                                   'distill_loss': kl_ce.item(), 'feature_loss': f_loss.item(),
                                   'epoch': epoch})
                step += 1

            wrapper.eval()
            total_val = 0.0
            count = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch['input_ids'].to(device_student)
                    mask = batch['attention_mask'].to(device_student)
                    labels = batch['labels'][:,1:].to(device_student).reshape(-1)
                    t_out = self.teacher_model(input_ids=batch['input_ids'].to(device_teacher),
                                               attention_mask=batch['attention_mask'].to(device_teacher),
                                               output_hidden_states=True)
                    t_logits = t_out.logits[:,:-1,:].reshape(-1, t_out.logits.size(-1)).to(device_student)
                    s_out = wrapper(input_ids=ids, attention_mask=mask)
                    s_logits = s_out.logits[:,:-1,:].reshape(-1, s_out.logits.size(-1))
                    kl_ce = self.logits_loss(s_logits, t_logits, labels, temperature, alpha)
                    student_feats = [s_out.hidden_states[i+1][:,:-1,:] for i in range(len(teacher_kept))]
                    teacher_feats = [t_out.hidden_states[i+1][:,:-1,:].to(device_student) for i in teacher_kept]
                    f_loss = self.feature_loss_selected_layers(student_feats, teacher_feats, wrapper.projectors, mask[:,1:].bool())
                    total_val += (kl_ce + gamma*f_loss).item()
                    count += 1
            avg_val = total_val / count
            val_loss_history.append(avg_val)
            if self.use_wandb:
                wandb.log({'val_loss': avg_val, 'epoch': epoch})
            wrapper.train()

        plt.figure()
        plt.plot(range(1, epochs+1), val_loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss over Epochs')
        plt.grid(True)
        plt.show()
        if self.use_wandb:
            wandb.log({ 'val_loss_plot': plt })
        return val_loss_history
