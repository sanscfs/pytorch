## Here we try to predict companies value based on their stocks on day 5 using data from days [1-4]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger

import lightning as L
torch.set_float32_matmul_precision("medium")
logger = TensorBoardLogger("lightning_logs", name="custom_lstm")

class LSTMbyHand(L.LightningModule):
    def __init__(self):
        super().__init__()
        ## Here we initialize random weights generator with normal distribution with mean 0 and std 1
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        ## Forget gate (% Long-term memory to remember, sigmoid)
        ### short-memory weight
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        ### input value weight
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## Input gate
        ### % Potential memory to remember (sigmoid)
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ### Potenial memory to remember (tanh)
        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## Output gate
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)

        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):

        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) +
                                              self.blr1)
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) +
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        update_long_memory = ((long_remember_percent * long_memory) +
                            (potential_remember_percent * potential_memory))
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                               (input_value * self.wo2) +
                               self.bo1)
        updated_short_memory = torch.tanh((output_percent * update_long_memory))

        return([update_long_memory, updated_short_memory])

    def forward(self, input):

        long_memory = 0
        short_memory = 0

        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.01)


    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2

        self.log("traing_loss :", loss) # log inhereted from lightning to store logs into ./lightning_logs

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss

def main():
    model = LSTMbyHand()
    ## First array is observed values for company A day 1-4, the second is for company B
    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.]) # What we want predict for day 5 company A and B
    dataset = TensorDataset(inputs, labels)
    ## Dataloader give us batching for data, shuffling data each epoch and use small fraction of data for quick train and debug
    dataloader = DataLoader(dataset, num_workers=0)

    print("Before optimization, the parameters are: ")
    for name, params in model.named_parameters():
        print(name, params.data)
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    trainer = L.Trainer(max_epochs=500, logger=logger)
    trainer.fit(model, train_dataloaders=dataloader)

    print("After training, the parameters are: ")
    for name, params in model.named_parameters():
        print(name, params.data)
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    ## Below we will continue training from the last checkpoint
    path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path # best_model_path refers to last checkpoint
    print("The new trainer will start where the last left off, and the ceck point data is here: " + path_to_best_checkpoint + "\n")
    trainer = L.Trainer(max_epochs=1000, logger=logger)
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)

    print("After post-training, the parameters are: ")
    for name, params in model.named_parameters():
        print(name, params.data)
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
if __name__ == "__main__":
    main()