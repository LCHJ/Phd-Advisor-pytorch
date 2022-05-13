import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)
x2 = np.linspace(0.635, 0.99, 30)
x3 = np.linspace(0, 1.1, 10)
x4 = np.linspace(0.001, 0.365, 30)

plt.figure(figsize=(11, 5), dpi=100)
plt.suptitle('Duality Analysis', fontsize=16)
plt.axis('equal')

ax = plt.subplot(1, 2, 1)
ax.set_title('y ~ min[-ln(1-y),1]', fontsize=14)
ax.plot(x3, np.clip(x3, 1, 1), color='g',
        linestyle='-.')
ax.plot(x, x, color='r',
        linestyle='-', label='y')
ax.plot(x, np.clip(-np.log(1 - x), 0, 1.0), color='b',
        linestyle='-', label='min[-ln(1-y),1]')
ax.plot(x2, np.clip(-np.log(1 - x2), 1, 1.5), color='b',
        linestyle=':')
ax.set_xlabel('y', fontsize=14)
ax.set_ylabel('Y', fontsize=14)

ax.legend(fontsize=14)

plt.ylim(0, 1.1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax2 = plt.subplot(1, 2, 2)
ax2.set_title('1-y ~ min[-ln(y),1)]', fontsize=14)
ax2.plot(x3, np.clip(x3, 1, 1), color='g',
         linestyle='-.')

ax2.plot(x, 1 - x, color='r',
         linestyle='-', label='1-y')
ax2.plot(x, np.clip(-np.log(x), 0, 1.0), color='b',
         linestyle='-', label='min[-ln(y),1)]')

ax2.plot(x4, np.clip(-np.log(x4), 0, 1.5), color='b',
         linestyle=':')

ax2.set_xlabel('y', fontsize=14)
ax2.set_ylabel('Y', fontsize=14)
ax2.legend(fontsize=14)

plt.ylim(0, 1.1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Duality Tactics y.jpg")
plt.show()
