import torch
from Dataset import construct_dataset
from Model import Fixed_Model, Proxy_Model, Trainer, Search, Final_Trainer, Proposed_Trainer
from Arichtecture import Network, FinalNetwork, Architect
from Arichtecture_single_path import Network_single_path, FinalNetwork_single_path, Architect_single_path

device = "cuda" if torch.cuda.is_available() else "cpu"

class Client():

    def __init__(
            self,
            data,
            appliance,
            lr_weight,
            lr_alpha,
            channel,
            node) -> None:

        self.data = data
        self.appliance = appliance
        self.train_dataloader, self.test_dataloader, self.scaler, self.input_dim, self.dataset, self.scalerx = construct_dataset(self.data, self.appliance)
        self.train_loss_fn = torch.nn.MSELoss(reduction="mean")
        self.lr_weight = lr_weight
        self.lr_alpha = lr_alpha
        self.channel = channel
        self.node = node
        
        self.local_model =  Fixed_Model(input_size=self.input_dim).to(device)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr_weight)
        self.local_trainer = Trainer(model=self.local_model, optimizer=self.local_optimizer, train_loss_fn=self.train_loss_fn)
        
        self.federated_model =  Fixed_Model(input_size=self.input_dim).to(device)
        self.federated_optimizer = torch.optim.Adam(self.federated_model.parameters(), lr=lr_weight)
        self.federated_trainer = Trainer(model=self.federated_model, optimizer=self.federated_optimizer, train_loss_fn=self.train_loss_fn)
        
        self.local_search_model = Network(C = self.channel, num_classes=self.input_dim, criterion=self.train_loss_fn, steps = self.node, multiplier=1).to(device)
        self.local_search_weight_optimizer = torch.optim.Adam(self.local_search_model.parameters(), lr=lr_weight)
        self.local_search_alpha_optimizer = torch.optim.Adam(self.local_search_model.arch_parameters(), lr=lr_alpha)
        self.local_architect = Architect(self.local_search_model, self.local_search_alpha_optimizer)
        self.local_search_trainer = Search(model=self.local_search_model, architecture=self.local_architect,
                                           search_weight_optimizer = self.local_search_weight_optimizer, 
                                            train_loss_fn=self.train_loss_fn)
        
        self.local_search_model_single = Network_single_path(C = self.channel, num_classes=self.input_dim, criterion=self.train_loss_fn, steps = self.node, multiplier=1).to(device)
        self.local_search_weight_optimizer_single = torch.optim.Adam(self.local_search_model_single.parameters(), lr=lr_weight)
        self.local_search_alpha_optimizer_single = torch.optim.Adam(self.local_search_model_single.arch_parameters(), lr=lr_alpha)
        self.local_architect_single = Architect_single_path(self.local_search_model_single, self.local_search_alpha_optimizer_single)
        self.local_search_trainer_single = Search(model=self.local_search_model_single, architecture=self.local_architect_single,
                                           search_weight_optimizer = self.local_search_weight_optimizer_single, 
                                            train_loss_fn=self.train_loss_fn)
        
    
    def local_train(self):
        self.local_trainer.trainer(self.train_dataloader)
        
    def fed_train(self):
        self.federated_trainer.trainer(self.train_dataloader)
    
    def fed_finetune(self):
        self.federated_trainer.fine_tune(self.train_dataloader)
    
    def get_fed_model(self):
        return self.federated_model
    
    def set_fed_model(self, model_params):
        self.federated_model.load_state_dict(model_params)
        
    def local_nas_search(self):
        loss = self.local_search_trainer.search(self.train_dataloader)
        return loss
        
    def local_nas_gen(self):
        self.best_gen = self.local_search_trainer.gen()
    
    def local_nas_final(self):
        self.local_final_model = FinalNetwork(self.best_gen, C=self.channel, num_output=self.input_dim).to(device)
        self.local_train_optimizer = torch.optim.Adam(self.local_final_model.parameters(), lr=self.lr_weight)  
        self.local_final_trainer = Final_Trainer(model=self.local_final_model, train_optimizer=self.local_train_optimizer, train_loss_fn=self.train_loss_fn)
        
    def local_nas_train(self):
        self.local_final_trainer.trainer(self.train_dataloader)
       
    def local_nas_search_single(self):
        loss = self.local_search_trainer_single.search(self.train_dataloader)
        return loss
        
    def local_nas_gen_single(self):
        self.best_gen_single = self.local_search_trainer_single.gen()
        return self.best_gen_single
    
    def local_nas_final_single(self):
        self.local_final_model_single = FinalNetwork_single_path(self.best_gen_single, C=self.channel, num_output=self.input_dim).to(device)
        self.local_train_optimizer_single = torch.optim.Adam(self.local_final_model_single.parameters(), lr=self.lr_weight)  
        self.local_final_trainer_single = Final_Trainer(model=self.local_final_model_single, train_optimizer=self.local_train_optimizer_single, train_loss_fn=self.train_loss_fn)
        
    def local_nas_train_single(self):
        self.local_final_trainer_single.trainer(self.train_dataloader)
        
    def fed_nas_final_single(self):
        self.fed_final_model_single = FinalNetwork_single_path(self.best_gen_single, C=self.channel, num_output=self.input_dim).to(device)
        self.fed_final_optimizer_single = torch.optim.Adam(self.fed_final_model_single.parameters(), lr=self.lr_weight)  
        self.fed_proxy_model_single = Proxy_Model(input_size=self.input_dim).to(device)
        self.fed_proxy_optimizer_single = torch.optim.Adam(self.fed_proxy_model_single.parameters(), lr=self.lr_weight)
        self.fed_trainer_single = Proposed_Trainer(final_model=self.fed_final_model_single, proxy_model=self.fed_proxy_model_single, 
                                                   final_optimizer=self.fed_final_optimizer_single, proxy_optimizer=self.fed_proxy_optimizer_single,
                                                   train_loss_fn=self.train_loss_fn)
        
    def fed_nas_train_single(self):
        self.fed_trainer_single.trainer(self.train_dataloader)
        
    def fed_nas_finetune_single(self):
        self.fed_trainer_single.fine_tune(self.train_dataloader)
        
    def get_fed_nas_model_single(self):
        return self.fed_proxy_model_single
    
    def set_fed_nas_model_single(self, model_params):
        self.fed_proxy_model_single.load_state_dict(model_params)