import torch
import numpy as np
from Model import Fixed_Model, Proxy_Model, Trainer
import pickle
import torch.utils.data as data
from sklearn.metrics import mean_absolute_error

device = "cuda" if torch.cuda.is_available() else "cpu"

client_genes = []
def sum_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/np.sum(y_true)

def load_all_genes():
    with open('all_clients_genes.pkl', 'rb') as f:
        all_genes = pickle.load(f)
    return all_genes

class Server():

    def __init__(self,
                 clients,
                 train_rounds,
                 search_rounds,
                 finetune_rounds) -> None:
        
        self.clients = clients
        self.train_rounds = train_rounds
        self.search_rounds = search_rounds
        self.finetune_rounds = finetune_rounds
        self.train_loss_fn = torch.nn.MSELoss(reduction="mean")
        
    def cen_model_test(self, model):
        self.model = model
        aver_sae = float()
        aver_mae = float()
        total_sae = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            x_test = x_test.unsqueeze(1)
            with torch.no_grad():
                pred = self.model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            pred = np.maximum(pred, 0)
            y_test = scalery.inverse_transform(y_test)
            mae = mean_absolute_error(y_test, pred)
            # print(f"MAE for client {id} is {mae}")
            sae = sum_absolute_error(y_test, pred)
            # print(f"SAE for client {id} is {sae}")
            total_mae = total_mae + mae
            total_sae = total_sae + sae
        aver_sae = total_sae/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_sae, aver_mae
        
    def local_model_test(self):
        aver_sae = float()
        aver_mae = float()
        total_sae = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].local_model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            x_test = x_test.unsqueeze(1)
            with torch.no_grad():
                pred = self.clients[id].local_model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            pred = np.maximum(pred, 0)
            y_test = scalery.inverse_transform(y_test)     
            mae = mean_absolute_error(y_test, pred)
            # print(f"MAE for client {id} is {mae}")
            sae = sum_absolute_error(y_test, pred)
            # print(f"SAE for client {id} is {sae}")
            total_mae = total_mae + mae
            total_sae = total_sae + sae
        aver_sae = total_sae/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_sae, aver_mae 
    
    def federated_model_test(self):
        aver_sae = float()
        aver_mae = float()
        total_sae = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].federated_model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            x_test = x_test.unsqueeze(1)
            with torch.no_grad():
                pred = self.clients[id].federated_model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            pred = np.maximum(pred, 0)
            y_test = scalery.inverse_transform(y_test)     
            mae = mean_absolute_error(y_test, pred)
            # print(f"MAE for client {id} is {mae}")
            sae = sum_absolute_error(y_test, pred)
            # print(f"SAE for client {id} is {sae}")
            total_mae = total_mae + mae
            total_sae = total_sae + sae
        aver_sae = total_sae/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_sae, aver_mae 
    
    def local_nas_model_test(self):
        aver_sae = float()
        aver_mae = float()
        total_sae = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].local_final_model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            with torch.no_grad():
                pred = self.clients[id].local_final_model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            pred = np.maximum(pred, 0)
            y_test = scalery.inverse_transform(y_test)    
            mae = mean_absolute_error(y_test, pred)
            # print(f"MAE for client {id} is {mae}")
            sae = sum_absolute_error(y_test, pred)
            # print(f"SAE for client {id} is {sae}")
            total_mae = total_mae + mae
            total_sae = total_sae + sae
        aver_sae = total_sae/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_sae, aver_mae 
    
    def local_single_nas_model_test(self):
        aver_sae = float()
        aver_mae = float()
        total_sae = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].local_final_model_single.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            with torch.no_grad():
                pred = self.clients[id].local_final_model_single(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            pred = np.maximum(pred, 0)
            y_test = scalery.inverse_transform(y_test)    
            mae = mean_absolute_error(y_test, pred)
            # print(f"MAE for client {id} is {mae}")
            sae = sum_absolute_error(y_test, pred)
            # print(f"SAE for client {id} is {sae}")
            total_mae = total_mae + mae
            total_sae = total_sae + sae
        aver_sae = total_sae/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_sae, aver_mae 
    
    def fed_single_nas_model_test(self):
        aver_sae = float()
        aver_mae = float()
        total_sae = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].fed_final_model_single.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            with torch.no_grad():
                pred = self.clients[id].fed_final_model_single(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            pred = np.maximum(pred, 0)
            y_test = scalery.inverse_transform(y_test)    
            mae = mean_absolute_error(y_test, pred)
            # print(f"MAE for client {id} is {mae}")
            sae = sum_absolute_error(y_test, pred)
            # print(f"SAE for client {id} is {sae}")
            total_mae = total_mae + mae
            total_sae = total_sae + sae
        aver_sae = total_sae/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_sae, aver_mae 
    
    def fed_model_average(self):
        models = []

        for client in self.clients:
            model = client.get_fed_model()
            models.append(model)
        
        avg_model_params = models[0].state_dict()

        for param_name in avg_model_params:
            for i in range(1, len(models)):
                avg_model_params[param_name] += models[i].state_dict()[param_name]
            avg_model_params[param_name] /= len(models)
        
        return avg_model_params
    
    def fed_single_nas_model_average(self):
        models = []

        for client in self.clients:
            model = client.get_fed_nas_model_single()
            models.append(model)
        
        avg_model_params = models[0].state_dict()

        for param_name in avg_model_params:
            for i in range(1, len(models)):
                avg_model_params[param_name] += models[i].state_dict()[param_name]
            avg_model_params[param_name] /= len(models)
        
        return avg_model_params
    
    def centralized_train(self):
        print('Centralized Training!')
        self.centralized_model = Fixed_Model(input_size=576).to(device)
        self.optimizer = torch.optim.Adam(self.centralized_model.parameters(), lr=5e-4)
        self.trainer = Trainer(model=self.centralized_model,
                                optimizer=self.optimizer, train_loss_fn=self.train_loss_fn)
        
        dataloaders = [client.dataset for client in self.clients]
        combined_dataset = data.ConcatDataset(dataloaders)
        combined_dataloader = data.DataLoader(combined_dataset, batch_size=64, shuffle=True, drop_last=True)
        # centralized_train_dataloader = merge_dataloaders(self.clients)
        for e in range(self.train_rounds):
            self.trainer.trainer(combined_dataloader)
            sae, mae = self.cen_model_test(self.centralized_model)
            print(f"rounds: {e+1} sae: {sae}, mae: {mae}")
            
    def local_train(self):
        print('Local Training!')
        a = 0
        for e in range(self.train_rounds):
            for client in self.clients:
                client.local_train()
            
            sae, mae = self.local_model_test()
            print(f"rounds: {e+1} sae: {sae}, mae: {mae}")

    
    def local_nas_search(self):
        print('NAS Searching!')
        for e in range(self.search_rounds):
            loss_avg = []
            for client in self.clients:
                loss = client.local_nas_search()
                loss_avg.append(loss)
            print(f"rounds: {e+1} loss: {np.mean(loss_avg)}")          

        for client in self.clients:
            client.local_nas_gen()    
            client.local_nas_final()
        
    def local_nas_train(self):
        print('NAS Training!')
        for e in range(self.train_rounds):
            for client in self.clients:
                client.local_nas_train()
            
            sae, mae = self.local_nas_model_test()
            print(f"rounds: {e+1} sae: {sae}, mae: {mae}")
    
    def local_nas_search_single(self):
        print('MNAS Searching!')
        for e in range(self.search_rounds):
            loss_avg = []
            for client in self.clients:
                loss = client.local_nas_search_single()
                loss_avg.append(loss)
            print(f"rounds: {e+1} loss: {np.mean(loss_avg)}")          

        for client in self.clients:
            self.gene = client.local_nas_gen_single()    
        
    def local_nas_train_single(self):
        print('MNAS Training!')
        for client in self.clients:
            client.local_nas_final_single()
        for e in range(self.train_rounds):
            for client in self.clients:
                client.local_nas_train_single()
            
            sae, mae = self.local_single_nas_model_test()
            print(f"rounds: {e+1} sae: {sae}, mae: {mae}")
            
    def fed_train(self):
        print('Federated Training!')
        self.fed_global_model = Fixed_Model(input_size=576).to(device)
        # local train
        for e in range(self.train_rounds):
            for client in self.clients:
                client.fed_train()

            # Model average   
            fed_global_model_params = self.fed_model_average()
            self.fed_global_model.load_state_dict(fed_global_model_params)

            # Model distribute
            for client in self.clients:
                client.set_fed_model(fed_global_model_params)

            sae, mae = self.federated_model_test()
            print(f"federated rounds: {e+1} sae: {sae}, mae: {mae}")
            
        for i in range(self.finetune_rounds):
            for client in self.clients:
                client.fed_finetune()

            sae, mae = self.federated_model_test()
            print(f"finetune rounds: {i+1} sae: {sae}, mae: {mae}")

    
    def fed_nas_train_single(self):
        print('Federated MNAS Training!')
        self.fed_global_model = Proxy_Model(input_size=576).to(device)
        for client in self.clients:
            client.fed_nas_final_single()
        # local train
        for e in range(self.train_rounds):
            for client in self.clients:
                client.fed_nas_train_single()

            # Model average   
            fed_global_model_params = self.fed_single_nas_model_average()
            self.fed_global_model.load_state_dict(fed_global_model_params)

            # Model distribute
            for client in self.clients:
                client.set_fed_nas_model_single(fed_global_model_params)

            sae, mae = self.fed_single_nas_model_test()
            print(f"federated rounds: {e+1} sae: {sae}, mae: {mae}")
            
        for i in range(self.finetune_rounds):
            for client in self.clients:
                client.fed_nas_finetune_single()

            sae, mae = self.fed_single_nas_model_test()
            print(f"finetune rounds: {i+1} sae: {sae}, mae: {mae}")
    
            
    