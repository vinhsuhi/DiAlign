import torch
import torch.nn as nn


class InnerProductWithWeightsAffinity(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InnerProductWithWeightsAffinity, self).__init__()
        self.d = output_dim
        self.A = nn.Linear(input_dim, output_dim)
        self.complex_lin = nn.Linear(output_dim * 2, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.complex_out = nn.Linear(output_dim, 1)

    def forward(self, X, Y, weights, complex=False):
        if complex:
            return self.forward_complex(X, Y, weights)
        coefficients = torch.tanh(self.A(weights)).unsqueeze(dim=1)
        res = (X * coefficients) @ Y.transpose(1, 2)
        #res = nn.functional.softplus(res) - 0.5
        return res
    
    def forward_complex(self, X, Y, weights):
        coefficients = torch.tanh(self.A(weights)).unsqueeze(dim=1)
        X_expanded = (X * coefficients).unsqueeze(2)
        Y_expanded = Y.unsqueeze(1)
        
        X_repeated = X_expanded.repeat(1, 1, Y.shape[1], 1)
        Y_repeated = Y_expanded.repeat(1, X.shape[1], 1, 1)
        
        U = torch.cat((X_repeated, Y_repeated), dim=3)
        U_out1 = self.complex_lin(U)
        U_out2 = self.dropout(self.act(U_out1))
        U_out3 = self.complex_out(U_out2)
        
        # import pdb; pdb.set_trace()
        
        return U_out3.squeeze(-1) # bs x m x n
        
    # def forward(self, Xs, Ys, Ws):
    #     return [self._forward(X, Y, W) for X, Y, W in zip(Xs, Ys, Ws)]

# class InnerProductWithWeightsAffinity(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(InnerProductWithWeightsAffinity, self).__init__()
#         self.d = output_dim
#         self.A = nn.Linear(input_dim, output_dim)

#     def forward(self, X, Y, weights):
#         '''
#         weights: global weight!
#         '''
#         # import pdb; pdb.set_trace()
#         # assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
#         coefficients = torch.tanh(self.A(weights)).unsqueeze(1)
#         #res = torch.matmul(X * coefficients, Y.transpose(0, 1))
#         res = (X * coefficients) @ Y.transpose(-2, -1)
#         #res = nn.functional.softplus(res) - 0.5
#         return res

#     # def forward(self, Xs, Ys, Ws):
#     #     return [self._forward(X, Y, W) for X, Y, W in zip(Xs, Ys, Ws)]
