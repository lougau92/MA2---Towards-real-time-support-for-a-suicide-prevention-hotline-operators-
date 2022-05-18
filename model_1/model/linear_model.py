class RegressionModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.dense_h1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu_h1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.dense_out = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X):

        out = self.relu_h1(self.dense_h1(X))
        out = self.dropout(out)
        out = self.dense_out(out)

        return out