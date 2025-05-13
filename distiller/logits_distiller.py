import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

class LogitsDistiller:
    def __init__(self, teacher_model, student_model, tokenizer, optimizer, scheduler,
                 use_wandb=False, wandb_key=None, project_name=None, run_name=None):
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

    def logits_loss(self, student_logits, teacher_logits, labels,
                    temperature=2.0, alpha=0.5):
        log_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        forward_kl = F.kl_div(log_student, soft_teacher,
                             reduction='batchmean') * (temperature ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)
        return alpha * ce_loss + (1 - alpha) * forward_kl

    def distill(self, train_loader, val_loader,
                device_teacher, device_student,
                epochs=1, grad_accumulation_steps=8,
                alpha=0.1, temperature=2.0):
        self.teacher_model.eval()
        self.student_model.train()

        if self.use_wandb:
            import wandb
            wandb.login(key=self.wandb_key)
            wandb.init(project=self.project_name,
                       name=self.run_name, reinit=True)

        val_history = []
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            step = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device_student)
                attention_mask = batch['attention_mask'].to(device_student)
                labels = batch['labels'][:, 1:].reshape(-1).to(device_student)

                with torch.no_grad():
                    t_out = self.teacher_model(
                        input_ids=batch['input_ids'].to(device_teacher),
                        attention_mask=batch['attention_mask'].to(device_teacher),
                        output_hidden_states=False
                    )
                    t_logits = t_out.logits[:, :-1, :].reshape(-1, t_out.logits.size(-1)).to(device_student)

                s_out = self.student_model(input_ids=input_ids,
                                           attention_mask=attention_mask)
                s_logits = s_out.logits[:, :-1, :].reshape(-1, s_out.logits.size(-1))

                loss = self.logits_loss(s_logits, t_logits, labels,
                                        temperature=temperature, alpha=alpha)
                loss = loss / grad_accumulation_steps
                loss.backward()

                if (step + 1) % grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if self.use_wandb:
                        import wandb
                        wandb.log({'train_loss': loss.item() * grad_accumulation_steps,
                                   'epoch': epoch, 'step': global_step})
                    global_step += 1
                step += 1
                epoch_loss += loss.item()
                
            self.student_model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device_student)
                    attention_mask = batch['attention_mask'].to(device_student)
                    labels = batch['labels'][:, 1:].reshape(-1).to(device_student)

                    t_out = self.teacher_model(
                        input_ids=batch['input_ids'].to(device_teacher),
                        attention_mask=batch['attention_mask'].to(device_teacher)
                    )
                    t_logits = t_out.logits[:, :-1, :].reshape(-1, t_out.logits.size(-1)).to(device_student)

                    s_out = self.student_model(input_ids=input_ids,
                                               attention_mask=attention_mask)
                    s_logits = s_out.logits[:, :-1, :].reshape(-1, s_out.logits.size(-1))

                    loss_val = self.logits_loss(s_logits, t_logits, labels,
                                                temperature=temperature, alpha=alpha)
                    val_loss += loss_val.item()
                    val_steps += 1
            avg_val = val_loss / max(val_steps, 1)
            val_history.append(avg_val)
            if self.use_wandb:
                import wandb
                wandb.log({'val_loss': avg_val, 'epoch': epoch, 'step': global_step})
            self.student_model.train()

        plt.figure()
        plt.plot(range(1, epochs + 1), val_history, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Over Epochs')
        plt.grid(True)
        plt.show()

        return val_history
