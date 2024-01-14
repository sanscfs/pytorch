import torch
import torch.nn as nn # Weight and bias tensors
import torch.nn.functional as F # Activation functions

import lightning as L

class BasicNN(nn.Module): # Inherit class from nn.Module

    def __init__(self): # Initialization method for new class
        super().__init__() # Initialization method for the nn.Module
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # Create first weight as parameter for our neural network and disable optimizing by gradient descent
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input): # Forward pass method
        input_to_top_relu = input * self.w00 + self.b00 # First layer
        top_relu_output = F.relu(input_to_top_relu) # Second layer
        scaled_top_relu_output = top_relu_output * self.w01 # Third layer

        input_to_bottom_relu = input * self.w10 + self.b10 # First layer
        bottom_relu_output = F.relu(input_to_bottom_relu) # Second layer
        scaled_bottom_relu_output = bottom_relu_output * self.w11 # Third layer

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias # Third layer

        output = F.relu(input_to_final_relu) # Final layer

        return output

class BasicNNTrain(nn.Module): # Inherit class from nn.Module

    def __init__(self): # Initialization method for new class
        super().__init__() # Initialization method for the nn.Module
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # Create first weight as parameter for our neural network and disable optimizing by gradient descent
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True) # Create last weight as parameter for our neural network and enable optimizing by gradient descent

    def forward(self, input): # Forward pass method
        input_to_top_relu = input * self.w00 + self.b00 # First layer
        top_relu_output = F.relu(input_to_top_relu) # Second layer
        scaled_top_relu_output = top_relu_output * self.w01 # Third layer

        input_to_bottom_relu = input * self.w10 + self.b10 # First layer
        bottom_relu_output = F.relu(input_to_bottom_relu) # Second layer
        scaled_bottom_relu_output = bottom_relu_output * self.w11 # Third layer

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias # Third layer

        output = F.relu(input_to_final_relu) # Final layer

        return output

## Lightning

class BasicLightning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # Create first weight as parameter for our neural network and disable optimizing by gradient descent
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input): # Forward pass method
        input_to_top_relu = input * self.w00 + self.b00 # First layer
        top_relu_output = F.relu(input_to_top_relu) # Second layer
        scaled_top_relu_output = top_relu_output * self.w01 # Third layer

        input_to_bottom_relu = input * self.w10 + self.b10 # First layer
        bottom_relu_output = F.relu(input_to_bottom_relu) # Second layer
        scaled_bottom_relu_output = bottom_relu_output * self.w11 # Third layer

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias # Third layer

        output = F.relu(input_to_final_relu) # Final layer

        return output


class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # Create first weight as parameter for our neural network and disable optimizing by gradient descent
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True) # Create last weight as parameter for our neural network and enable optimizing by gradient descent

        self.learning_rate = 0.01

    def forward(self, input): # Forward pass method
        input_to_top_relu = input * self.w00 + self.b00 # First layer
        top_relu_output = F.relu(input_to_top_relu) # Second layer
        scaled_top_relu_output = top_relu_output * self.w01 # Third layer

        input_to_bottom_relu = input * self.w10 + self.b10 # First layer
        bottom_relu_output = F.relu(input_to_bottom_relu) # Second layer
        scaled_bottom_relu_output = bottom_relu_output * self.w11 # Third layer

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias # Third layer

        output = F.relu(input_to_final_relu) # Final layer

        return output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, labels_i = batch
        output_i = self(input_i)
        loss = (output_i - labels_i)**2

        return loss