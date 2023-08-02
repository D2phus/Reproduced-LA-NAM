import torch
import torch.nn as nn
import torch.optim as optim

def train(config, 
          model: nn.Module, 
          dataloader_train: torch.utils.data.DataLoader, 
          dataloader_val: torch.utils.data.DataLoader,
         ):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) 
        
        criterion = nn.MSELoss() if config.regression else nn.CrossEntropyLoss()
        
        losses_train = []
        losses_val = []
        for epoch in range(config.num_epochs):
            loss_train = train_epoch(criterion, model, dataloader_train, optimizer)
            loss_val = evaluate_epoch(criterion, model, dataloader_val)
            # save statistics
            losses_train.append(loss_train.detach().cpu().numpy().item())
            losses_val.append(loss_val.detach().cpu().numpy().item())
                
            # print statistics
            if epoch % config.log_loss_frequency == 0: 
                print(f"=============EPOCH {epoch+1}==============")
                print(f"loss_train: {(loss_train.detach().cpu().numpy().item()): .3f}, loss_val: {(loss_val.detach().cpu().numpy().item()): .3f}")
                
        print("Finished Training.")
        return sum(losses_train)
            

def train_epoch(criterion, 
                model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Adam):
    model.train()
    avg_loss = 0.0
    for batch in dataloader:
        features, targets = batch

        optimizer.zero_grad()

        out, fnn = model(features)

        step_loss = criterion(out, targets)

        step_loss.backward()
        optimizer.step()

        avg_loss += step_loss
        
    return avg_loss / len(dataloader)
    
def  evaluate_epoch(criterion, 
                    model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
) -> torch.Tensor: 
    """
    Perform an epoch of evaluation on dataloader 
    """
    model.eval()
    avg_loss = 0.0
    for batch in dataloader:
                # Accumulates loss in dataset.
        with torch.no_grad():
            features, targets = batch
    
            out, fnn = model(features)

            step_loss = criterion(out, targets)
            avg_loss += step_loss

    return avg_loss / len(dataloader)
