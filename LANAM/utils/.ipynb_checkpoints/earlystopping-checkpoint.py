import torch 
import torch.nn as nn 

class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=10, delta=0): 
        self.patience = patience
        self.delta = delta
        self.counter = 0 
        self.best_val_loss = float('inf')
    
    def early_stop(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.best_val_loss + self.delta): 
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False