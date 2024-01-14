from torch import tensor
from torch.utils.data import TensorDataset, DataLoader

def generate_dataset():
    inputs = tensor([[1., 0., 0., 0.], # Troll2
                        [0., 1., 0., 0.], # is
                        [0., 0., 1., 0.], # great
                        [0., 0., 0., 1.]]) # Gymkata

    print(inputs)

    labels = tensor([[0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                        [0., 1., 0., 0.]])

    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)
    return dataloader

def generate_vocab():
    vocab = {'Troll2': 0,
             'is': 1,
             'great': 2,
             'Gymkata': 3}
    return vocab