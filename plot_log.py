from config import (
    RESULT_PATH
)

import numpy as np
from matplotlib import pyplot as plt

object_name = "rc-car_test"

with open(RESULT_PATH(object_name) / "log.txt", "r") as f:
    log = f.readlines()

log = log[2:((len(log)-2)//3)*3+2]
y_train = list(map(float, log[1::3]))
y_val = list(map(float, log[2::3]))
x = np.arange(1, len(y_train)+1)

plt.plot(x,y_train, label='train', c='tab:blue')
plt.plot(x,y_val, label='validation', c='tab:orange')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
y_max = max(max(y_train), max(y_val)) + 0.1
y_min = min(min(y_train), min(y_val)) - 0.1
x_max = x[-1]+1
x_min = x[0]
plt.yticks(np.arange(y_min,y_max,0.05))
plt.xticks(np.arange(x_min,x_max,1))
plt.show()