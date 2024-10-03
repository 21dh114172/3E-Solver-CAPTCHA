import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def get_random_samples(dataloader, amount=5):
    dataset = dataloader.dataset
    
    random_indexes = []
    for i in range(amount):
        random_indexes.append(int(np.random.random()*len(dataset)))
    
    samples = []
    for index in random_indexes:
        samples.append(dataset[index])
    return samples

def save_model(epochs, model, optimizer, criterion, vocab, id2token, image_height, image_width, model_name="final_model.pth"):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'vocab': vocab,
                'id2token': id2token,
                'image_height': image_height,
                'image_width': image_width,
                }, f'result/{model_name}')
    torch.save(model, f'result/embed_{model_name}')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, vocab, id2token, image_height, image_width, model_name="best_model.pth"
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'vocab': vocab,
                'id2token': id2token,
                'image_height': image_height,
                'image_width': image_width,
                }, f'result/{model_name}')
            torch.save(model, f'result/embed_{model_name}')
            
def sigmoid_rampup(current, rampup_length):
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
        

def get_current_consistency_weight(consistency, consistency_rampup, epoch):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
    

class Seq2SeqLoss(nn.Module):
    def __init__(self):
        super(Seq2SeqLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, y):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        y: [batch_size, max_len]
        """
        max_len = y.size(1)

        loss = sum([self.criterion(outputs[:, i, :], y[:, i + 1]) for i in range(max_len - 1)]) / (max_len - 1)

        return loss


class ConsistentLoss_MT(nn.Module):
    def __init__(self):
        super(ConsistentLoss_MT, self).__init__()

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        max_len = outputs.size(1) + 1

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            loss += F.mse_loss(input_softmax, target_softmax, reduction='none').mean()

        return loss / (max_len - 1)


class ConsistentLoss_MT_Temperature(nn.Module):
    def __init__(self, threshold):
        super(ConsistentLoss_MT_Temperature, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        max_len = outputs.size(1) + 1

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            max_probs, targets_u = torch.max(target_softmax, dim=-1)
            mask = max_probs.ge(self.threshold).float()

            loss += (F.mse_loss(input_softmax, target_softmax, reduction='none').mean(dim=-1) * mask).mean()

        return loss / (max_len - 1)

class ConsistentLoss(nn.Module):
    def __init__(self, threshold):
        super(ConsistentLoss, self).__init__()

        self.threshold = threshold

    def compute_consistent_loss(self, logits_u_w, logits_u_s):
        """
        outputs: [secondary_batch_size, max_len-1, vocab_size]
        outputs_ema: [secondary_batch_size, max_len-1, vocab_size]
        """
        max_len = logits_u_s.size(1) + 1

        pseudo_label = torch.softmax(logits_u_w, dim=-1)
        loss_all = 0
        for i in range(max_len - 1):
            max_probs, targets_u = torch.max(pseudo_label[:, i, :], dim=-1)
            mask = max_probs.ge(self.threshold).float()
            loss_all += (F.cross_entropy(logits_u_s[:, i, :], targets_u,
                                         reduction='none') * mask).mean()

        return loss_all / (max_len - 1)

    def forward(self, logits_u_w, logits_u_s):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        loss = self.compute_consistent_loss(logits_u_w, logits_u_s)

        return loss


def compute_seq_acc(outputs, y, max_len):
    """
    outputs: [batch_size, max_len-1, vocab_size]
    y: [batch_size, max_len]
    """

    accuracy_clevel, accuracy_all = compute_acc_step(outputs, y, max_len)

    return accuracy_clevel, accuracy_all


def compute_acc_step(outputs, y, max_len):
    num_eq = (y[:, 1:].data == outputs.max(2)[1]).sum(dim=1)
    accuracy_clevel = num_eq.sum() / (max_len - 1)
    accuracy_all = (num_eq == max_len - 1).sum()

    return accuracy_clevel.item(), accuracy_all.item()

