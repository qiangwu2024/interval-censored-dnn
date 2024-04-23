import torch
from torch import nn
import numpy as np

#%% -------------------------
def LFD(Zi_train,Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U,Lambda_V,g_train,Beta,n_layer,n_node,n_lr,n_epoch):
    Zi_train = torch.Tensor(Zi_train)
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    U_train = torch.Tensor(U_train)
    V_train = torch.Tensor(V_train)
    De1_train = torch.Tensor(De1_train)
    De2_train = torch.Tensor(De2_train)
    De3_train = torch.Tensor(De3_train)
    Lambda_U = torch.Tensor(Lambda_U)
    Lambda_V = torch.Tensor(Lambda_V)
    X_U = torch.Tensor(np.c_[X_train, U_train, V_train])
    Beta = torch.Tensor(Beta)
    d = X_train.size()[1]
    # ----------------------------
    class DNNAB(torch.nn.Module):
        def __init__(self):
            super(DNNAB, self).__init__()
            layers = []
            layers.append(nn.Linear(d+2, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 2))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred


    # ----------------------------
    model = DNNAB()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def Loss(De1,De2,De3,Zi,Z,Beta,Lambda_U,Lambda_V,g_X,a_b):
        f_U = Lambda_U * torch.exp(torch.matmul(Z, Beta) + g_X)
        f_V = Lambda_V * torch.exp(torch.matmul(Z, Beta) + g_X)
        Ef_U = torch.exp(- f_U)
        Ef_V = torch.exp(- f_V)
        Q1_y = Ef_U/(1-Ef_U+1e-3)
        Q2_y = Ef_U/(Ef_U-Ef_V+1e-3)
        Q3_y = Ef_V/(Ef_U-Ef_V+1e-3)
        Es = f_U*(De1*Q1_y-De2*Q2_y)*(Zi-a_b[:,0]) + f_V*(De2*Q3_y-De3)*(Zi-a_b[:,1])
        Loss_f = torch.mean(Es**2)
        return Loss_f


    # -----------------------------
    for epoch in range(n_epoch):
        pred_ab = model(X_U)
        loss = Loss(De1_train,De2_train,De3_train,Zi_train,Z_train,Beta,Lambda_U,Lambda_V,g_train,pred_ab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#%% ----------------------
    ab_train = model(X_U)
    ab_train = ab_train.detach().numpy()
    return ab_train