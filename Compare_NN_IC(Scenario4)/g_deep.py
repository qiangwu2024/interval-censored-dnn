# %%
import torch
from torch import nn
#%% --------------------------
def g_D(train_data,Z_validation,Z_test,Lambda_U,Lambda_V,n_layer,n_node,n_lr,n_epoch,lambda_param,k_param):
    Z_train = torch.Tensor(train_data['Z'])
    U_train = torch.Tensor(train_data['U'])
    V_train = torch.Tensor(train_data['V'])
    De1_train = torch.Tensor(train_data['De1'])
    De2_train = torch.Tensor(train_data['De2'])
    De3_train = torch.Tensor(train_data['De3'])
    g_train_true = torch.Tensor(train_data['g_Z'])
    Z_test = torch.Tensor(Z_test)
    Z_validation = torch.Tensor(Z_validation)
    Lambda_U = torch.Tensor(Lambda_U)
    Lambda_V = torch.Tensor(Lambda_V)
    Z_dim = Z_train.size()[1]
    # ----------------------------
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(Z_dim, n_node))
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


    def my_loss(De1,De2,De3,Lambda_U,Lambda_V,g_Z):
        assert(len(g_Z.shape) == len(De1.shape))
        assert(len(g_Z.shape) == len(Lambda_U.shape))
        Ezg = torch.exp(g_Z)
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- Lambda_U * Ezg) + 1e-5) + De2 * torch.log(torch.exp(- Lambda_U * Ezg) - torch.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return loss_fun

    true_Lambda_U = (lambda_param*U_train)**k_param
    true_Lambda_V = (lambda_param*V_train)**k_param
    best_loss = my_loss(De1_train,De2_train,De3_train,true_Lambda_U,true_Lambda_V,g_train_true)
    
    # ----------------------------
    for epoch in range(n_epoch):
        pred_g_Z = model(Z_train)
        loss = my_loss(De1_train,De2_train,De3_train,Lambda_U,Lambda_V,pred_g_Z[:, 0])
        if (loss <= best_loss):
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # %% ----------------------
    g_train = model(Z_train)
    g_validation = model(Z_validation)
    g_test = model(Z_test)
    g_train = g_train[:,0].detach().numpy()
    g_validation = g_validation[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_validation': g_validation,
        'g_test': g_test
    }