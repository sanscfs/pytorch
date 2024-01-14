import lightning as L
import torch.nn as nn # Weight and bias tensors
from torch.optim import Adam # Fit data with backpropagation
from torch import tensor

from utils.plot import plot_embedding_results, create_dataframe_linear
from utils.dataloader import generate_dataset, generate_vocab
from tests.embedding_test import run_embedding_test

class WordEmbeddingWithLinear(L.LightningModule):
    def __init__(self):
        super().__init__()
        ## Instead of creating each parameter (weight) with nn.Parameter
        ## we can use nn.Linear with only 2 calls: from input nodes and to output nodes
        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        hidden = self.input_to_hidden(input)
        output = self.hidden_to_output(hidden)

        return output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output = self.forward(input_i)
        loss = self.loss(output, label_i)

        return loss

def main():
    dataloader=generate_dataset()
    vocab=generate_vocab()
    model = WordEmbeddingWithLinear()
  
    plot_embedding_results(create_dataframe_linear(model))

    ## Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloaders=dataloader)

    plot_embedding_results(create_dataframe_linear(model))

    run_embedding_test(model, dim=1)

    word_embedings = nn.Embedding.from_pretrained(model.input_to_hidden.weight.T) # .T means transpose rows into columns, and vise-versa
    print(word_embedings.weight)

    ## Now we can connect those embeddings to huge neural networks!
    print("Troll2 embeddings: ", word_embedings(tensor(vocab['Troll2'])))
    print("is embeddings: ", word_embedings(tensor(vocab['is'])))
    print("great embeddings: ", word_embedings(tensor(vocab['great'])))
    print("Gymkata embeddings: ", word_embedings(tensor(vocab['Gymkata'])))
if __name__ == "__main__":
    main()