# This source is coded with great reference to
# facebookresearch/EGG/egg/zoo/compo_vs_generalization/
# (https://github.com/facebookresearch/EGG/tree/main/egg/zoo/compo_vs_generalization),
# which are licensed under the MIT license
# (https://github.com/facebookresearch/EGG/blob/main/LICENSE).

import torch
import egg.core as core
import json
import time

def check_cuda():
    if torch.cuda.is_available():
        print("yes")
    else:
        print("no")

class Timer():
    def __init__(self, task_name: str):
        self.task_name = task_name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *_ex):
        diff = time.time() - self.start
        print(f"task: {self.task_name}", flush=True)
        print(f"time: {diff}s", flush=True)

class Evaluator(core.Callback):
    def __init__(self, loaders, device, freq=1):
        self.loaders = loaders
        self.device = device
        self.epoch = 0
        self.freq = freq
        self.results = {}
    
    def evaluate(self):
        game = self.trainer.game
        game.eval()

        # temporarily evacuated
        old_loss = game.loss

        for loader_name, loader, metric in self.loaders:
            acc_or, acc = 0.0, 0.0
            n_batches = 0
            game.loss = metric

            for batch in loader:
                n_batches += 1
                batch = core.move_to(batch, self.device)
                with torch.no_grad():
                    _, rest = game(*batch)
                acc += rest['acc']
                acc_or += rest['acc_or']
            
            self.results[loader_name] = {
                'acc': acc / n_batches,
                'acc_or': acc_or / n_batches,
            }
        
        self.results['epoch'] = self.epoch
        output = json.dumps(self.results)
        print(output, flush=True)

        # reset
        game.loss = old_loss
        game.train()
    
    def on_train_end(self):
        self.evaluate()
    
    def on_epoch_end(self, *_stuff):
        self.epoch += 1
        if self.freq <= 0 or self.epoch % self.freq != 0:
            return
        self.evaluate()
