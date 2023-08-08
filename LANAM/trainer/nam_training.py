import torch
import torch.nn as nn
import torch.optim as optim
from nam.trainer.losses import penalized_loss

def mae(logits, targets):
    return ((logits.view(-1) - targets.view(-1)).abs().sum() / logits.numel()).item()


def accuracy(logits, targets):
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()

def nam_train(config, 
          model: nn.Module, 
          dataloader_train: torch.utils.data.DataLoader, 
          dataloader_val: torch.utils.data.DataLoader,
         ):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) 
        
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        
        metrics_name = "MAE" if config.regression else "Accuracy"
        metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        
        losses_train = list()
        losses_val = list()
        metricses_train = list()
        metricses_val = list()
        for epoch in range(config.num_epochs):
            loss_train, metrics_train = train_epoch(criterion, metrics, optimizer, model, dataloader_train)
            loss_val, metrics_val = evaluate_epoch(criterion, model, dataloader_val)
            # save statistics
            losses_train.append(loss_train.detach().cpu().numpy().item())
            losses_val.append(loss_val.detach().cpu().numpy().item())
            metricses_train.append(metrics_train)
            metricses_val.append(metrics_val)
                    
            # print statistics
            if epoch % config.log_loss_frequency == 0: 
                print(f"=============EPOCH {epoch+1}==============")
                print(f"loss_train_epoch: {loss_train.detach().cpu().numpy().item()}, {metrics_name}_train_epoch: {metrics_train}")
                print(f"loss_val_epoch: {loss_val.detach().cpu().numpy().item()}, {metrics_name}_val_epoch: {metrics_val}")
                
        print("Finished Training.")
        return losses_train[-1], metricses_train[-1]
            
def train_epoch(
    criterion, 
    metrics, 
    optimizer: torch.optim.Adam, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    scheduler=None, 
) -> torch.Tensor: 
    """
    Perform an epoch of gradient-descent optimization on dataloader 
    """
    model.train()
    avg_loss = 0.0
    avg_metrics = 0.0
            
    for batch in dataloader:
        features, targets = batch

        optimizer.zero_grad()

        preds, fnn_out = model(features)

        step_loss = criterion(preds, fnn_out, model, targets)
        step_metrics = metrics(preds, targets)

        step_loss.backward()
        optimizer.step()
        
        avg_loss += step_loss
        avg_metrics += step_metrics
        
        if scheduler is not None: 
            scheduler.step()
                
    return avg_loss / len(dataloader), avg_metrics / len(dataloader)


def  evaluate_epoch(
    criterion, 
    metrics, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader):
    """
    Perform an epoch of evaluation on dataloader 
    """
    model.eval()
    avg_loss = 0.0
    avg_metrics = 0.0
    for batch in dataloader:
                # Accumulates loss in dataset.
        with torch.no_grad():
            features, targets = batch
    
            preds, fnn_out = model(features)

            step_loss = criterion(preds, fnn_out, model, targets)
            step_metrics = metrics(preds, targets)
            avg_loss += step_loss
            avg_metrics += step_metrics

    return avg_loss / len(dataloader), avg_metrics / len(dataloader)

