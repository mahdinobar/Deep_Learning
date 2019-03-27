import torch
from losses import LossMSE
from activation_functions import Relu, Tanh
from neural_network import Linear, Sequential
from torch.autograd import Variable


# ------------------------------------------------------------------------------------------
# Compare Loss implementation
# ------------------------------------------------------------------------------------------
print('\n--------------> Check Loss')
loss = LossMSE()
loss_torch = torch.nn.MSELoss()

a_var = Variable(torch.empty(5, 5).normal_(), requires_grad=True)
b_var = Variable(torch.empty(5, 5).normal_(), requires_grad=True)

# Forward
print('Forward loss \n', loss.forward(a_var, b_var))
print('Forward loss pytorch \n', loss_torch(a_var, b_var))

# Backward
print('Backward loss\n', loss.backward())
loss_torch(a_var, b_var).backward()
print('Backward loss pytorch \n', a_var.grad.data)


# ------------------------------------------------------------------------------------------
# Compare Activation function implementation
# ------------------------------------------------------------------------------------------
print('\n--------------> Check Activation Functions')
print('ReLU')
relu = Relu()
relu_pytorch = torch.nn.ReLU()

output_forward = relu.forward(a_var)
print('ReLU Forward:\n', output_forward)
output_forward_pytorch = relu_pytorch(a_var)
print('ReLU pytorch forward \n', output_forward_pytorch)

#TODO check Backward pass
print('ReLU backward \n', relu.backward(output_forward))
output_forward_pytorch.backward(output_forward)
print('ReLU pytorch backward \n', a_var.grad.data)

print('\nTanh')
th = Tanh()
output_forward = th.forward(a_var)
print('Tanh Forward:\n', output_forward)
output_forward_pytorch = torch.tanh(a_var)
print('Tanh pytorch forward \n', output_forward_pytorch)
#TODO check Backward pass
print('Tanh backward \n', th.backward(output_forward))
output_forward_pytorch.backward(output_forward)
print('Tanh pytorch backward \n', a_var.grad.data)



# ------------------------------------------------------------------------------------------
# Compare Linear implementation
# ------------------------------------------------------------------------------------------
print('\n--------------> Check Linear')
in_dim = 3
out_dim = 5

input_tensor = torch.empty(10, in_dim).normal_()
output_tensor = torch.empty(10, out_dim).normal_()

init_w = torch.empty(out_dim, in_dim).normal_()
init_b = torch.empty(1, out_dim).normal_()
init = [init_w, init_b]

model = Linear(input_dim=in_dim, output_dim=out_dim, w0=init_w, b0=init_b)
model_pytorch = torch.nn.Linear(3, 5)

print('Parameter model')
for i, param in enumerate(model.param()):
    print(param[0])

print('Parameter model pytorch')
for i, param in enumerate(model_pytorch.parameters()):
    param.data = init[i]
    print(param)

print('\nLinear layer')
print('Parameter model update')
model.forward(input_tensor)
model.backward(output_tensor)
for i, param in enumerate(model.param()):
    print(param[0])

print('\nLinear layer pytorch')
print('Parameter model pytorch update')
input_tensor = Variable(input_tensor, requires_grad=True)
forward_torch = model_pytorch(input_tensor)
forward_torch.backward(output_tensor)
for i, param in enumerate(model_pytorch.parameters()):
    print(param)

# ------------------------------------------------------------------------------------------
# Compare Sequential implementation
# ------------------------------------------------------------------------------------------
print('\n--------------> Check Sequential')

input_tensor = torch.empty(10, 3).normal_()
input_tensor = Variable(input_tensor, requires_grad=True)

model = Sequential(Relu(), Tanh())
model_pytorch = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Tanh())

# Forward
output_tensor = model.forward(input_tensor)
print('output model: \n', output_tensor)
output_tensor_pytorch = model_pytorch(input_tensor)
print('output pytorch model: \n', output_tensor_pytorch)

# Backward
print('backward \n', model.backward(output_tensor))
output_tensor_pytorch.backward(output_tensor)
print('backward \n', input_tensor.grad.data)

#TODO check Sequential with linear