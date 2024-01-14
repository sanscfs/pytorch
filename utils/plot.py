import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns # Samual Norman Seaborn, The West Wing
import pandas as pd

def plot_results(input, output, filename):
    plt.figure()
    sns.set(style="whitegrid")
    sns.lineplot(x=input, 
                y=output.detach(), # NOTE: we call detach() because final_bias has a gradient
                color='green', 
                linewidth=2.5)
    plt.ylabel('Output')
    plt.xlabel('Input')
    plt.savefig(filename)

def plot_embedding_results(dataframe):
    plt.figure()
    sns.scatterplot(x="w1", y="w2", hue="token", data=dataframe)
    for i in range(len(dataframe)):
        plt.text(dataframe.w1[i], dataframe.w2[i], dataframe.token[i],
                horizontalalignment='left',
                size='large',
                color='black',
                weight='bold')
    plt.show()

def create_dataframe_linear(model):
    data = {
        "w1": model.input_to_hidden.weight.detach().numpy()[0], # detach() to remove gradient from the tensors
        "w2": model.input_to_hidden.weight.detach().numpy()[1],
        "token": ["Troll2", "is", "great", "Gymkata"],
        "input": ["input1", "input2", "input3", "input4"]
    }
    df = pd.DataFrame(data)
    return df