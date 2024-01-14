import torch
import torch.nn as nn

from torch.optim import Adam # Fit data with backpropagation
from torch.distributions.uniform import Uniform # Initialize weights in network
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot import plot_embedding_results
from utils.dataloader import generate_dataset
from tests.embedding_test import run_embedding_test
## One-hot encodings

dataloader=generate_dataset()

class WordEmbeddingFromScratch(L.LightningModule):
    def __init__(self):
        ## Create and initialize Weight tensor, and create the loss function
        super().__init__()
        L.seed_everything(seed=42)
        min_val = -0.5
        max_val = 0.5

        ### Uniform distribute [-0.5, 0.5] among the 4 inputs
        self.input1_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_val, max_val).sample())
        
        self.input2_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        self.input3_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        self.input4_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        ### Uniform distribute [-0.5, 0.5] among the 4 outputs
        self.output1_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        self.output2_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        self.output3_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        self.output4_w1 = nn.Parameter(Uniform(min_val, max_val).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_val, max_val).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        ## Make a forward pas through the network
        input = input[0] # Input delivered inside of a list with double brackets [[...]], so we will remove extra pair

        ### One-hot encodings multiplied by weights
        inputs_to_top_hidden = ((input[0] * self.input1_w1) +
                                (input[1] * self.input2_w1) +
                                (input[2] * self.input3_w1) +
                                (input[3] * self.input4_w1))

        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) +
                                   (input[1] * self.input2_w2) +
                                   (input[2] * self.input3_w2) +
                                   (input[3] * self.input4_w2))

        ### Top hidden layer; multuply activation function results by weights, sum and save to variable
        output1 = ((inputs_to_top_hidden * self.output1_w1) +
                   (inputs_to_bottom_hidden * self.output1_w2))
        
        output2 = ((inputs_to_top_hidden * self.output2_w1) +
                   (inputs_to_bottom_hidden * self.output2_w2))

        output3 = ((inputs_to_top_hidden * self.output3_w1) +
                   (inputs_to_bottom_hidden * self.output3_w2))

        output4 = ((inputs_to_top_hidden * self.output4_w1) +
                   (inputs_to_bottom_hidden * self.output4_w2))

        output_presoftmax = torch.stack([output1, output2, output3, output4]) # use torch.stack to preserve the gradients

        return(output_presoftmax)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        # Calculate loss
        input_i, label_i = batch # Get the first element of the input and label
        output_i = self.forward(input_i) # Pass the input to the neural network
        loss = self.loss(output_i, label_i[0])
        return loss

def main():
    model = WordEmbeddingFromScratch()
    print("Before optimization, the parameters are...")
    ## Put weight values into a dictionary for readability...
    data = {
        "w1": [model.input1_w1.item(), # Activation function weights on top
               model.input2_w1.item(),
               model.input3_w1.item(),
               model.input4_w1.item()],
        "w2": [model.input1_w2.item(), # Activation function weights on bottom
               model.input2_w2.item(),
               model.input3_w2.item(),
               model.input4_w2.item()],
        "token": ["Troll2", "is", "great", "Gymkata"], # Input tokens
        "input": ["input1", "input2", "input3", "input4"]
    }

    ## print data to view embeddings
    df = pd.DataFrame(data)
    df

    ## plot data to view embeddings on graph
    plot_embedding_results(df)

    for name, param in model.named_parameters():
        print(name, param.data)

    trainer = L.Trainer(max_epochs=100)
    ## pass embedding model and labeled data to trainer
    trainer.fit(model, train_dataloaders=dataloader)
    data = {
        "w1": [model.input1_w1.item(), # Activation function weights on top
               model.input2_w1.item(),
               model.input3_w1.item(),
               model.input4_w1.item()],
        "w2": [model.input1_w2.item(), # Activation function weights on bottom
               model.input2_w2.item(),
               model.input3_w2.item(),
               model.input4_w2.item()],
        "token": ["Troll2", "is", "great", "Gymkata"], # Input tokens
        "input": ["input1", "input2", "input3", "input4"]
    }

    ## print data to view embeddings
    df = pd.DataFrame(data) 
    print(df)
    plot_embedding_results(df)

    run_embedding_test(model, dim=0)

if __name__ == "__main__":
    main()