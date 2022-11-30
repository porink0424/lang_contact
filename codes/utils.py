import torch
import egg.core as core
import json

def check_cuda():
    if torch.cuda.is_available():
        print("yes")
    else:
        print("no")

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

        game.loss = old_loss
        game.train()
    
    def on_train_end(self):
        self.evaluate()
    
    def on_epoch_end(self, *_stuff):
        self.epoch += 1
        if self.freq <= 0 or self.epoch % self.freq != 0:
            return
        self.evaluate()