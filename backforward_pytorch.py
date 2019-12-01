import net_pytorch
import torch.nn as nn
from torch.autograd import Variable
import torch as t
import torch.optim as optim
net = net_pytorch.Net()
# for name,parameters in net.named_parameters():
#     print(name,parameters.size())
input = Variable(t.randn(1,1,32,32))
output = net(input)
target = Variable(t.arange(0,10))
target = target.float()
criterion = nn.MSELoss()
loss = criterion(output,target)
#print(loss)
net.zero_grad()
print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)
optimizer = optim.SGD(net.parameters(),lr = 0.01)
optimizer.zero_grad()
optimizer.step()

# output.backward(Variable(t.ones(1,10)))
# print(output.size())






