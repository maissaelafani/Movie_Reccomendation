import numpy as np
import torch
import matplotlib.pyplot as plt

#namesngenre = torch.from_numpy(np.load("namesngenre.npy"))
ratings_test = torch.from_numpy(np.load("ratings_test.npy"))
ratings_train = torch.from_numpy(np.load("ratings_train.npy"))

n_pers = ratings_train.shape[0]
n_film = ratings_train.shape[1]

# Hyperparameters
k = 1 
lmda = 0
mu = 0

R = ratings_train
Corrector = torch.ones(R.shape)
Corrector[torch.isnan(R)] = 0
R = torch.nan_to_num(R)

R_test = ratings_test
Corrector_test = torch.ones(R_test.shape)
Corrector_test[torch.isnan(R_test)] = 0
R_test = torch.nan_to_num(R_test)

def dist(A,B,Corrector):
    return torch.sum(torch.mul(torch.square(A-B),Corrector))
    #return torch.sum(torch.nan_to_num(torch.square(A-B)))

def cost(I,U):
    return dist(R,torch.matmul(I,torch.transpose(U,0,1)),Corrector) + lmda*torch.linalg.norm(I) + mu*torch.linalg.norm(U)
    
def cost_test(I,U):
    return dist(R_test,torch.matmul(I,torch.transpose(U,0,1)),Corrector_test)

I = torch.rand((n_pers,k),requires_grad=True)
U = torch.rand((n_film,k),requires_grad=True)


C = cost(I,U)

cost_test_evo = []
cost_train_evo = []

lr = 0.0005
for i in range(200):
    C = cost(I,U)
        
    cost_test_evo.append(cost_test(I,U).item())
    cost_train_evo.append(C.item())
    
    grad = torch.autograd.grad(C,(I,U))
    I = I - lr*(grad[0])
    U = U - lr*(grad[1])

print(cost_test(I,U).item(),"   ",C.item())

plt.plot(range(200),cost_train_evo)
plt.plot(range(200),cost_test_evo)
plt.show()

    
