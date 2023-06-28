from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

model = MLP(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

epoch = 20
for k in range(epoch):
  # forward pass
  ypred = [model(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

  model.zero_grad()
  
  # backward pass
  loss.backward()
  
  # update
  learning_rate = 0.3 - 0.25*k/(epoch-1)
  for p in model.parameters():
    p.data -= learning_rate * p.grad
  
  print(k, loss.data)

print(ypred)