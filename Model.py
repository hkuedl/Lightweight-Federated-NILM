from torch import nn
import torch
import torch.utils.data as data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Proxy_Model(nn.Module):
    def __init__(self,
                input_size):
        super(Proxy_Model, self).__init__()
        self.n = 32
        self.conv1 = nn.Conv1d(1, self.n, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(self.n, self.n, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(self.n, self.n, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n * 576, 1024)
        self.fc2 = nn.Linear(1024, input_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1, self.n * 576)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Fixed_Model(nn.Module):
    def __init__(self,
                input_size):
        super(Fixed_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 576, 1024)
        self.fc2 = nn.Linear(1024, input_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = x.view(-1, 32 * 576)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
       
class Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loss_fn) -> None:

        self.model = model
        self.train_loss_fn = train_loss_fn
        self.optimizer = optimizer
    
    def trainer(
            self,
            train_dataloader: data.DataLoader):
        
        self.model.train()
        for X, Y in train_dataloader:
            running_loss = 0.0
            X = X.unsqueeze(1)
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(X)
            loss = self.train_loss_fn(Y, pred)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        # print(f'Loss: {running_loss / (32)}')
    
    def fine_tune(
            self,
            train_dataloader: data.DataLoader):
        
        self.model.train()
        for X, Y in train_dataloader:
            X = X.unsqueeze(1)
            # Forward pass
            pred = self.model(X)
            loss = self.train_loss_fn(Y, pred)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class Search:
    def __init__(self,
                 model,
                 architecture,
                 search_weight_optimizer: torch.optim.Optimizer,
                 train_loss_fn
                 ) -> None:

        self.model = model
        self.search_weight_optimizer = search_weight_optimizer
        self.train_loss_fn = train_loss_fn
        self.architecture = architecture
    
    def search(
            self,
            train_dataloader: data.DataLoader):
        
        self.model.train()
        for X, Y in train_dataloader:
            search_loss = 0.0
            self.architecture.step(X, Y)
            self.search_weight_optimizer.zero_grad()
            pred = self.model(X)
            loss = self.train_loss_fn(Y, pred)          
            # Update weight
            loss.backward()
            self.search_weight_optimizer.step()
            search_loss += loss.item()
        return search_loss
        
    def gen(self):
        gene = self.model.genotype()
        print(gene)
        return gene
        
    
            
class Final_Trainer:
    def __init__(self,
                 model,
                 train_loss_fn,
                 train_optimizer: torch.optim.Optimizer
                 ) -> None:

        self.model = model
        self.train_optimizer = train_optimizer
        self.train_loss_fn = train_loss_fn
        
    def trainer(
            self,
            train_dataloader: data.DataLoader):
        self.model.train()
        for X, Y in train_dataloader:
            train_loss = 0.0
            self.train_optimizer.zero_grad()
            # Forward pass
            pred = self.model(X)
            loss = self.train_loss_fn(Y, pred)
            # Backward pass
            loss.backward()
            self.train_optimizer.step()
            train_loss += loss.item()
        
    def fine_tune(
            self,
            train_dataloader: data.DataLoader):
        
        self.model.train()
        for X, Y in train_dataloader:
            # Forward pass
            pred = self.model(X)
            loss = self.train_loss_fn(Y, pred)
            
            # Backward pass
            self.train_optimizer.zero_grad()
            loss.backward()
            self.train_optimizer.step()

class Proposed_Trainer:
    def __init__(self,
                 final_model: nn.Module,
                 proxy_model: nn.Module,
                 final_optimizer: torch.optim.Optimizer,
                 proxy_optimizer: torch.optim.Optimizer,
                 train_loss_fn) -> None:

        self.final_model = final_model
        self.proxy_model = proxy_model
        self.final_optimizer = final_optimizer
        self.proxy_optimizer = proxy_optimizer
        self.train_loss_fn = train_loss_fn
    
    def trainer(
            self,
            train_dataloader: data.DataLoader):
        
        self.final_model.train()
        self.proxy_model.train()
        for X, Y in train_dataloader:
            
            self.final_optimizer.zero_grad()
            self.proxy_optimizer.zero_grad()
            
            # Forward pass
            pred_final = self.final_model(X)
            pred_proxy = self.proxy_model(X)
            
            # Label loss
            loss_final = self.train_loss_fn(Y, pred_final)
            loss_proxy = self.train_loss_fn(Y, pred_proxy)
            sum_loss = loss_final + loss_proxy
                  
            # Knowledge distillation loss
            distillation_loss = self.train_loss_fn(pred_final, pred_proxy)
            
            # Combine losses
            loss_final_combined = loss_final.clone() + distillation_loss
            loss_proxy_combined = loss_proxy.clone() + distillation_loss
            
            # Backward pass
            loss_final_combined.backward(retain_graph=True)
            loss_proxy_combined.backward()
            
            self.final_optimizer.step()
            self.proxy_optimizer.step()
        
        
    def fine_tune(
            self,
            train_dataloader: data.DataLoader):
        
        self.final_model.train()
        for X, Y in train_dataloader:
            
            # Forward pass
            pred = self.final_model(X)
            loss = self.train_loss_fn(Y, pred)
            
            # Backward pass
            self.final_optimizer.zero_grad()
            loss.backward()
            self.final_optimizer.step()