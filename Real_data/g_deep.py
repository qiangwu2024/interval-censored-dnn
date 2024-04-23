import torch
from torch import nn
#%% --------------------------
def g_D(Z_train,X_train,De1_train,De2_train,De3_train,X_test,X_sort1,X_sort0,Lambda_U,Lambda_V,Beta0,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    De1_train = torch.Tensor(De1_train)
    De2_train = torch.Tensor(De2_train)
    De3_train = torch.Tensor(De3_train)
    X_test = torch.Tensor(X_test)
    X_sort1 = torch.Tensor(X_sort1)
    X_sort0 = torch.Tensor(X_sort0)
    Lambda_U = torch.Tensor(Lambda_U)
    Lambda_V = torch.Tensor(Lambda_V)
    Beta0 = torch.Tensor(Beta0)
    d = X_train.size()[1]
    # -------------------------
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
    # ---------------------------
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)
    def my_loss(De1,De2,De3,Z,Beta,Lambda_U,Lambda_V,g_X):
        assert(len(g_X.shape) == len(De1.shape))
        assert(len(g_X.shape) == len(Lambda_U.shape))
        assert(len(g_X.shape) == len(Lambda_V.shape))
        Ezg = torch.exp(torch.matmul(Z, Beta) + g_X)
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- Lambda_U * Ezg) + 1e-5) + De2 * torch.log(torch.exp(- Lambda_U * Ezg) - torch.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return loss_fun
    # -------------------------
    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(De1_train,De2_train,De3_train,Z_train,Beta0,Lambda_U,Lambda_V,pred_g_X[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    g_train = model(X_train)
    g_test = model(X_test)
    g_test1 = model(X_sort1)
    g_test0 = model(X_sort0)
    g_train = g_train[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    g_test1 = g_test1[:,0].detach().numpy()
    g_test0 = g_test0[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test,
        'g_test1': g_test1,
        'g_test0': g_test0
    }