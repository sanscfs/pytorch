import torch
from torch.optim import SGD # Stochastic Gradient Descent

def train(model, inputs, labels, lr):
    for epoch in range(100):

        optimizer = SGD(model.parameters(), lr) # Create an optimizer
        total_loss = 0

        for iteration in range(len(inputs)):

            input_i = inputs[iteration]
            label_i = labels[iteration]
            output_i = model(input_i)

            loss = (output_i - label_i)**2

            loss.backward()

            total_loss += float(loss)

        if (total_loss < 0.0001):
            print("Num steps: " + str(epoch))
            break

        optimizer.step() # Take a step toward the optimal value.
        optimizer.zero_grad() # This zeroes out the gradient stored in "model". 

        print("Step: " + str(epoch) + " Final bias: " + str(model.final_bias.data) + "\n")
    print("Total loss: " + str(total_loss))
    print("Final bias, after optimization: " + str(model.final_bias.data))