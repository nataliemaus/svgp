import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_model(
    model,
    mll,
    train_x,
    train_y,
    learning_rte=0.01,
    n_epochs=10,
    train_bsz=128,
    max_norm=2.0,
    verbose=True,
):
    if len(train_y.shape) > 1:
        train_y = train_y.squeeze()
    model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    train_bsz = min(len(train_y),train_bsz)
    train_dataset = TensorDataset(train_x.to(device), train_y.to(device))
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for e in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = -mll(output, scores.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            if verbose:
                print(f"Loss after {e + 1} epochs: {loss.item()}")
    model.eval()

    return model