import numpy as np
import matplotlib.pyplot as plt   # package for plotting    

X1 = np.random.rand(100)
X2 = np.random.rand(100)
Y = 0.2*X1 + 0.3*X2 + 0.5

def pred_Y(pred, Y):
	fig = plt.figure(figsize=(10,12))
	ax = fig.add_subplot(211, projection='3d')
	ax.scatter(X1, X2, Y)
	ax.scatter(X1, X2, pred)
	fig.add_subplot(ax)
	plt.show()
	ax.remove()  # delete ax1 from the figure
	

W1 = np.random.uniform(-1, 1)
W2 = np.random.uniform(-1, 1)
b =  np.random.uniform(-1, 1)

lr = 0.07

for epoch in range(1000):
    pred = W1*X1 + W2*X2 + b
    error = np.abs(pred-Y).mean()
    if error < 0.001:
        break
    grad_W1 = lr*((pred-Y)*X1).mean()
    grad_W2 = lr*((pred-Y)*X2).mean()
    grad_b = lr*(pred-Y).mean()
    W1 -= grad_W1
    W2 -= grad_W2
    b -= grad_b
    if epoch%100 == 0:
        pred_Y(pred, Y)
        print(f'epoch: {epoch}, error: {error}')

	
	
	
	
	