

class WarmupCosineAnnealingLR:
        def __init__(self, optimizer, warmup_epochs=3, warmup_factor=0.1, T_max=40, eta_min=1e-6):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.warmup_factor = warmup_factor
            self.initial_lr = optimizer.param_groups[0]['lr']
            self.current_epoch = 0
            self.T_max = T_max
            self.eta_min = eta_min
            
            # Set initial warmup learning rate
            self._set_warmup_lr(0)
        
        def _set_warmup_lr(self, epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                warmup_lr = self.initial_lr * (self.warmup_factor + 
                                              (1 - self.warmup_factor) * epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
        
        def _set_cosine_lr(self, epoch):
            # Adjusted epoch to account for warmup
            adjusted_epoch = epoch - self.warmup_epochs
            # Cosine annealing formula
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(adjusted_epoch * torch.pi / (self.T_max - self.warmup_epochs))))
            lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        def step(self, metrics=None):
            self.current_epoch += 1
            
            if self.current_epoch <= self.warmup_epochs:
                self._set_warmup_lr(self.current_epoch)
            else:
                self._set_cosine_lr(self.current_epoch)
        
        def get_last_lr(self):
            return [param_group['lr'] for param_group in self.optimizer.param_groups]