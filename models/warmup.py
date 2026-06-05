# SPDX-License-Identifier: MIT

class LRWarmupScheduler:
    """
    Learning rate warmup scheduler that linearly increases LR from 0 to base LR.
    
    This scheduler is designed to work with PyTorch optimizers and should be called
    at each training iteration during the warmup phase.
    
    Args:
        optimizer: PyTorch optimizer instance
        warmup_steps: Number of iterations to warmup (default: 500)
        base_lr: Target learning rate to reach after warmup (default: taken from optimizer)
    
    Usage:
        warmup_scheduler = LRWarmupScheduler(optimizer, warmup_steps=500)
        for step in range(total_steps):
            if step < warmup_steps:
                warmup_scheduler.step(step)
            # ... training code ...
    """
    def __init__(self, optimizer, warmup_steps=500, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        
        # Get base LR from optimizer if not provided
        if base_lr is None:
            self.base_lr = optimizer.param_groups[0]['lr']
        else:
            self.base_lr = base_lr
            
        # Store initial LR for all param groups
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, current_step):
        """
        Update learning rate based on current step.
        
        Args:
            current_step: Current training iteration (0-indexed)
        """
        if current_step >= self.warmup_steps:
            # Warmup complete, return without changing LR
            return
        
        # Linear warmup: lr = base_lr * (current_step / warmup_steps)
        # Start from a small value to avoid lr=0 at step 0
        warmup_factor = (current_step + 1) / self.warmup_steps
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * warmup_factor
