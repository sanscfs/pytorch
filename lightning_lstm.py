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

class LightningLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        ## We have only 1 feature for each company (DAY), therefore we use input size=1
        ## We have only 1 output for each company (prediction for day 5), therefor we use hidden_size=1
        ## If we want to plug outputs to another NN, we can use different hidden size (f.e. for simple nn)
        L.seed_everything(seed=42)
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)


    def forward(self, input):

        ## Transpose input values from company for days 1-4 from being in a row to being in a singe column with view()
        ## len(input) has 4 values, so view 1 column per value (1 feature (day) = 1 column) 
        input_trans = input.view(len(input), 1)

        ## lstm_out contains short-term memory, therefore it has 4 values, e.g. 1 value for each unrolled lstm
        lstm_out, temp = self.lstm(input_trans)

        ## Exctract last item in the lstm_out
        prediction = lstm_out[-1]
        return prediction


    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)


    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2

        self.log("traing_loss :", loss) # log inhereted from lightning to store logs into ./lightning_logs

        if (label_i == 0):
            self.log("company_A", output_i)
        else:
            self.log("company_B", output_i)

        return loss

def main():
    model = LightningLSTM()
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

    trainer = L.Trainer(max_epochs=200, logger=logger, log_every_n_steps=2)
    trainer.fit(model, train_dataloaders=dataloader)

    print("After training, the parameters are: ")
    for name, params in model.named_parameters():
        print(name, params.data)
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

    ## Below we will continue training from the last checkpoint
    path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path # best_model_path refers to last checkpoint
    trainer = L.Trainer(max_epochs=1200, logger=logger, log_every_n_steps=2)
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)

    print("After post-training, the parameters are: ")
    for name, params in model.named_parameters():
        print(name, params.data)
    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
if __name__ == "__main__":
    main()