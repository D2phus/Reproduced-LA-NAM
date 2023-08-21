import torch
import torch.nn as nn
import torch.optim as optim
 
def train(config, 
          model: nn.Module, 
          dataloader_train: torch.utils.data.DataLoader, 
          dataloader_val: torch.utils.data.DataLoader,
         ):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) 
        
        criterion = nn.MSELoss(reduction='sum') if config.likelihood == 'regression' else nn.CrossEntropyLoss(reduction='sum')

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
                avg_loss_train = loss_train.detach().cpu().numpy().item() / len(dataloader_train.dataset)
                avg_loss_val = loss_val.detach().cpu().numpy().item() / len(dataloader_val.dataset)
                print(f"loss_train: {avg_loss_train: .3f}, loss_val: {avg_loss_val: .3f}")
                
        print("Finished Training.")
        return losses_train[-1]
            
def train_epoch(criterion, 
                model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Adam):
    model.train()
    loss = 0.0
    for batch in dataloader:
        features, targets = batch

        optimizer.zero_grad()

        outs, _ = model(features)
        step_loss = criterion(outs.reshape(-1, 1), targets)
            
        step_loss.backward()
        optimizer.step()

        loss += step_loss
        
    return loss
    
def  evaluate_epoch(criterion, 
                    model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
) -> torch.Tensor: 
    """
    Perform an epoch of evaluation on dataloader 
    """
    model.eval()
    loss = 0.0
    for batch in dataloader:
                # Accumulates loss in dataset.
        with torch.no_grad():
            features, targets = batch
    
            outs, _ = model(features)
            step_loss = criterion(outs, targets)

            loss += step_loss

    return loss
