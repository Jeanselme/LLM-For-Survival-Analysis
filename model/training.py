from transformers import Trainer
import torch
    
class DeepHitTrainer(Trainer):
    """
        Can be understand as a one layer neural network with a final softmax layer
        Only the loss needs to be updated as some points do not have labels and present a different loss
    """
    def __init__(self, model, args, data_collator = None, train_dataset = None, eval_dataset = None, tokenizer = None, model_init = None):
        super().__init__(model, args, data_collator = data_collator, train_dataset = train_dataset, eval_dataset = eval_dataset, tokenizer = tokenizer, model_init = model_init)
        (e, t) = train_dataset.labels
        self.max = t.max()
        self.splits = torch.linspace(0, self.max, len(self.model.splits))

    def binarise(self, t):
        return torch.bucketize(torch.clamp(t, 0, self.max), self.splits)
 
    def compute_loss(self, model, inputs, return_outputs = False):
        """
            Compute the nll for DeepHit
        """
        # Get labels
        labels = inputs.pop("labels")
        e, t = labels[:, 0], labels[:, 1]
        t = self.binarise(t)

        outputs = model(**inputs)

        # Apply softmax to use BERT Multi Class
        logits = torch.softmax(outputs.get('logits').view(-1, len(self.model.splits)), dim = 1)

        # Uncensored
        loss = torch.sum(torch.log(logits[e == 1][:, t[e == 1]] + 1e-10)) # Instanteneous risk

        # Censored loss
        cifs = logits[e == 0].cumsum(dim = 1) # Cumulative incidence functions
        loss += torch.sum(torch.log(1 - cifs[:, t[e == 0]] + 1e-10))

        # Average
        loss = - loss / len(t)

        return (loss, outputs) if return_outputs else loss
    
    def predict(self, data):
        output = super().predict(data)
        return torch.softmax(torch.tensor(output.predictions), dim = 1).detach().numpy().cumsum(1)