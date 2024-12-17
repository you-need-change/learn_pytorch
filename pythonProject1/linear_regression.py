#   1. y_^
#   2. loss
#   3. backward
#   4. 更新

import torch
import torch.nn.functional as F

x_data = torch.Tensor([[-0.29, 0.49, 0.18, -0.29, 0.00, 0.00, -0.53, -0.03],
                       [-0.88, -0.15, 0.08, -0.41, 0.00, -0.21, -0.77, -0.67],
                       [-0.06, 0.84, 0.05, 0.00, 0.00, -0.31, -0.49, -0.63],
                       [-0.88, -0.11, 0.08, -0.54, -0.78, -0.16, -0.92, 0.00],
                       [0.00, 0.38, -0.34, -0.29, -0.60, 0.28, 0.89, -0.60],
                       [-0.41, 0.17, 0.21, 0.00, 0.00, -0.24, -0.89, -0.70],
                       [-0.65, -0.22, -0.18, -0.35, -0.79, -0.08, -0.85, -0.83],
                       [0.18, 0.16, 0.00, 0.00, 0.00, 0.05, -0.95, -0.73],
                       [-0.76, 0.98, 0.15, -0.09, 0.28, -0.09, -0.93, 0.07],
                       [-0.06, 0.26, 0.57, 0.00, 0.00, 0.00, -0.87, 0.10]])
y_data = torch.Tensor([[0], [1], [0], [1], [0], [1], [0], [1], [0], [0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = F.sigmoid(self.linear(x))
        return x
model = LinearModel()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)