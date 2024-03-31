import numpy as np
import matplotlib.pyplot as plt   # package for plotting    

s = np.random.uniform(-2,0,1000)

count, bins, _ = plt.hist(s, 15, density=True)
# print(np.sum(count*np.diff(bins)))   # 1.0

# 기준선
plt.plot(bins, np.ones_like(bins), linewidth=1, color='r')
plt.show()
