import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def get_volt():
    """Measure voltage."""
    noise = np.random.normal(0, 4)  # v: measurement noise.
    volt_mean = 14.4            # volt_mean: mean (nominal) voltage [V].
    volt = volt_mean + noise   # volt_meas: measured voltage [V] (observable).
    return volt

def avg_filter(n, x_sample, x_avg):
    """Calculate average voltage using a average filter."""
    alpha = (n - 1) / n
    x_avg = (1 - alpha) * x_sample + alpha * x_avg
    return x_avg

# Input parameters.
time_end = 10
dt = 0.2

time = np.arange(0, time_end, dt)
n_samples = len(time)
x_measure_save = np.zeros(n_samples)
x_avg_save = np.zeros(n_samples)

x_avg = 0
for i in range(n_samples):
    n = i + 1
    measure_volt = get_volt()
    x_avg = avg_filter(n, measure_volt, x_avg)

    x_measure_save[i] = measure_volt
    x_avg_save[i] = x_avg
    
plt.plot(time, x_measure_save, 'r*', label='Measured')
plt.plot(time, x_avg_save, 'b-', label='Average')
plt.legend(loc='upper left')
plt.title('Measured Voltages v.s. Average Filter Values')
plt.xlabel('Time [sec]')
plt.ylabel('Volt [V]')

# t-x plot
plt.show()
# plt.savefig('average_filter.png')