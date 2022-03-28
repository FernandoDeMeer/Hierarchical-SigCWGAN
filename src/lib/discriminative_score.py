# Adapted from: https://github.com/d9n13lt4n/timegan-pytorch
import torch
from tqdm import tqdm, trange
import numpy as np

class DiscriminativeScoreDataset(torch.utils.data.Dataset):
    r"""Torch Dataset for predicting the discriminative score of a dataset
    Args:
    - real_data, generated_data (np.ndarray): the datasets to be trained on (Batch, Time, Dim)
    """
    def __init__(self, real_data, gen_data):

        self.X = torch.FloatTensor(torch.cat((real_data,gen_data)))
        self.Y = torch.FloatTensor(torch.cat((torch.ones(real_data.shape[0]),torch.zeros(gen_data.shape[0]))))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]

class ClassificationRNN(torch.nn.Module):
    r"""A general RNN model for time-series classification
    """
    def __init__(self, args):
        super(ClassificationRNN, self).__init__()
        self.model_type = args['model_type']

        self.input_size = args['in_dim']
        self.hidden_size = args['h_dim']
        self.output_size = args['out_dim']
        self.num_layers = 2

        self.padding_value = args['padding_value']
        self.seq_len = args['seq_len']

        self.rnn_module = self._get_rnn_module(self.model_type)

        self.rnn_layer = self.rnn_module(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.linear_layer = torch.nn.Linear(
            in_features=self.hidden_size * self.seq_len,
            out_features=self.output_size,
        )

    def _get_rnn_module(self, model_type):
        if model_type == "rnn":
            return torch.nn.RNN
        elif model_type == "LSTM":
            return torch.nn.LSTM
        elif model_type == "gru":
            return torch.nn.GRU

    def forward(self, X):
        # Dynamic RNN input for ignoring paddings
        H_o, H_t = self.rnn_layer(X)
        H_o = torch.reshape(H_o,(H_o.shape[0],-1))
        logits = self.linear_layer(H_o)
        logits = torch.sigmoid(logits)
        return logits

def discriminative_score(real_train_data, gen_train_data,real_test_data, gen_test_data):
    """
    Args:
    - (real/gen)_train_data (train_data, train_time): training time-series
    - (real/gen)_test_data (test_data, test_data): testing time-series

    Returns:
    - Disc_score: np.abs(0.5 - accuracy)
    """

    # Parameters
    no, seq_len, dim = real_train_data.shape

    # Set model parameters

    args = {}
    args["device"] = "cpu"
    args["task"] = "classification"
    args["model_type"] = "LSTM"
    args["bidirectional"] = False
    args["epochs"] = 500
    args["batch_size"] = 64
    args["in_dim"] = dim
    args["h_dim"] = dim
    args["out_dim"] = 2
    args["n_layers"] = 2
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["seq_len"] = real_train_data.shape[1]
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Output initialization
    perf = list()

    # Set training features and labels
    train_dataset = DiscriminativeScoreDataset(
        real_train_data, gen_train_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataset = DiscriminativeScoreDataset(
        real_test_data, gen_test_data)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=no,
        shuffle=False
    )

    # Initialize model
    model = ClassificationRNN(args)
    model.to(args["device"])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["learning_rate"]
    )
    print('Calculating discriminative score')
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x, train_y in train_dataloader:
            one_hot = torch.zeros(train_y.shape[0], 2)
            one_hot[torch.arange(train_y.shape[0]), train_y.long()] = 1

            # train_x = train_x.to(args["device"])
            # train_y = train_y.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x)
            loss = criterion(train_p, one_hot)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")


    # Evaluate the trained model
    with torch.no_grad():
        temp_perf = 0
        for test_x, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x,).cpu().numpy()
            test_p = np.argmax(test_p, axis= 1)

            test_p = np.round(np.reshape(test_p, [-1]),1)
            test_y = np.reshape(test_y.numpy(), [-1])

            temp_perf = np.average(test_p == test_y)

    perf.append(np.round(np.abs(0.5-temp_perf),4))

    return perf

