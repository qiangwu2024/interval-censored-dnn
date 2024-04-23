# %%
import torch
from torch import nn
#%% --------------------------
def g_D(train_data,X_test,Lambda_U,Lambda_V,Beta,Beta0,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    X_train = torch.Tensor(train_data['X'])
    U_train = torch.Tensor(train_data['U'])
    V_train = torch.Tensor(train_data['V'])
    De1_train = torch.Tensor(train_data['De1'])
    De2_train = torch.Tensor(train_data['De2'])
    De3_train = torch.Tensor(train_data['De3'])
    g_train_true = torch.Tensor(train_data['g_X'])
    X_test = torch.Tensor(X_test)
    Lambda_U = torch.Tensor(Lambda_U)
    Lambda_V = torch.Tensor(Lambda_V)
    Beta0 = torch.Tensor(Beta0)
    d = X_train.size()[1]
    # ---------------------------
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(d, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred


    # ----------------------------
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De1,De2,De3,Z,Beta,Lambda_U,Lambda_V,g_X):
        assert(len(g_X.shape) == len(De1.shape))
        assert(len(g_X.shape) == len(Z.shape))
        assert(len(g_X.shape) == len(Lambda_U.shape))
        Ezg = torch.exp(Z * Beta + g_X)
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- Lambda_U * Ezg) + 1e-5) + De2 * torch.log(torch.exp(- Lambda_U * Ezg) - torch.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return loss_fun


    # ----------------------------
    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(De1_train,De2_train,De3_train,Z_train,Beta0,Lambda_U,Lambda_V,pred_g_X[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # %% ----------------------
    # true_Lambda_U = torch.log(1 + U_train**(4/5)/6)
    # true_Lambda_V = torch.log(1 + V_train**(4/5)/6)
    # best_loss = my_loss(De1_train,De2_train,De3_train,Z_train,Beta,true_Lambda_U,true_Lambda_V,g_train_true)
    
    
    g_train = model(X_train)
    g_test = model(X_test)
    g_train = g_train[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test
    }