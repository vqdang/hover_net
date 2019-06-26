
def exp_decay_lr_schedule(nr_epochs, init_lr, decay_factor, decay_steps):
    lr_sched = []
    for i in range(1, nr_epochs):
        if i % decay_steps == 0:
            decay = decay_factor ** (i / decay_steps)
            new_lr = init_lr * decay 
            lr_sched.append((i, new_lr))
    return lr_sched
    