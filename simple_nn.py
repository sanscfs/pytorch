import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD # Stochastic Gradient Descent

from basicnn import * # Import the basicnn.py file
from utils.plot import plot_results
from utils.train import train

import lightning as L
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning import Trainer


def main():
    input = torch.linspace(start=0, end=1, steps=11) # Create a tensor of 100 values from 0 to 1

    print("Trained data:")
    model = BasicNN() # Create a new instance of our neural network
    output = model(input) # Pass the input tensor to the neural network
    for name, param in model.named_parameters():
        print(name, param.data)
    plot_results(input, output, './artifacts/BasicNN.pdf')

    print("\n Untrained data:")
    model = BasicNNTrain() 
    output = model(input)
    print("Untrained data output: " + str(output) + "\n")
    for name, param in model.named_parameters():
        print(name, param.data)
    plot_results(input, output, './artifacts/BasicNNUntrain.pdf')

    print("\n Data for training:")
    inputs = torch.tensor([0., 0.5, 1.])
    labels = torch.tensor([0., 1., 0.])
    print("Final bias, before training: " + str(model.final_bias.data) + "\n")
    train(model, inputs, labels, 0.1) # Train the model
    output_values = model(input)
    print("Trained data output: " + str(output_values) + "\n")
    plot_results(input, output_values, './artifacts/BasicNNTrain.pdf')

    model = BasicLightning()
    output_values = model(input)
    plot_results(input, output_values, './artifacts/BasicLightning.pdf')

    model = BasicLightningTrain()
    output_values = model(input)
    trainer = Trainer()
    tuner = Tuner(trainer)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=1)
    lr_find_results = tuner.lr_find(model,
                                            train_dataloaders=dataloader,
                                            num_training=100,
                                            min_lr=0.001,
                                            max_lr=1.0,
                                            early_stop_threshold=None)
    new_lr = lr_find_results.suggestion()
    print(f"lr_rind() suggest {new_lr:.5f} for the learning rate.")
    plot_results(input, output_values, './artifacts/BasicLightningTrain.pdf')

if __name__ == "__main__":
    main()