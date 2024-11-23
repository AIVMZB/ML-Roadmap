import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dataclasses import dataclass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResults:
    model: nn.Module
    train_losses: list
    val_losses: list


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        # Your task here is to build the MLP architecture which can fit the data
        # To add linear layer use nn.Linear(input_dim, output_dim)
        # To add activation function use nn.ReLU() or nn.Sigmoid() or other if you find it useful or interesting
        ## YOUR CODE STARTS HERE ##
        self.net = nn.Sequential(
            ...
        )
        ## YOUR CODE ENDS HERE ##
        
    def forward(self, x):
        return self.net(x)


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    """
    Returns train and validation dataloaders
    """
    data = pd.read_csv("diabetes.csv")
    x = data.drop("Outcome", axis=1).values
    y = data["Outcome"].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader


def train() -> TrainResults:
    ## specify correct value for input_dim ##
    input_dim = ...
    model = MLP(input_dim).to(DEVICE)
    criterion = nn.BCELoss() # The binary cross entropy loss is used
    
    ## Specify the suitable value for learning rate ##
    lr = ...
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader = get_dataloaders()
    train_loader = train_loader
    val_loader = val_loader

    # Training loop
    ## Specify the number of epochs ##
    num_epochs = ...

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs
            targets = targets
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    return TrainResults(model, train_losses, val_losses)


def plot_losses(train_loss: list, val_loss: list):
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.title("Losses plot")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # The code will run train process and plot train and validation loss. 
    # Build the model and set the hyper parameters. 
    # Try to avoid overfit
    # Good luck!!! I am sure you will succeed!!!
     
    train_results = train()
    plot_losses(train_results.train_losses, train_results.val_losses)

    model = train_results.model
    torch.save(model.state_dict(), "mlp.pth")
