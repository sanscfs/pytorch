from torch.nn import Softmax
from torch import tensor, round

def run_embedding_test(model, dim):
    softmax = Softmax(dim) # dim=0 so that we can apply it to rows of output values; if we set dim=1 then we would apply it to columns of values

    ## pass Troll2 as a one-hot encoded tensor into model, and run the output values through softmax
    print("Predict next word after Troll2: ", round(softmax(model(tensor([[1., 0., 0., 0.]]))), decimals=2)) # Predict word after Troll2
    print("Predict next word after is: ", round(softmax(model(tensor([[0., 1., 0., 0.]]))), decimals=2))
    print("Predict next word after great: ", round(softmax(model(tensor([[0., 0., 1., 0.]]))), decimals=2))
    print("Predict next word after Gymkata: ", round(softmax(model(tensor([[0., 0., 0., 1]]))), decimals=2))
