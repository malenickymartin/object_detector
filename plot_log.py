from config import (
    MODEL_PATH
)

import numpy as np
from matplotlib import pyplot as plt

object_name = "rc-car"

with open(MODEL_PATH(object_name) / "log.txt", "r") as f:
    log = f.readlines()

log = log[2:((len(log)-2)//3)*3-1]
y_train = list(map(float, log[1::3]))
y_val = list(map(float, log[2::3]))
x = np.arange(1, len(y_train)+1)

plt.plot(x,y_train, label='train', c='tab:blue')
plt.plot(x,y_val, label='validation', c='tab:orange')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.yticks(np.arange(0,1,0.05))
plt.show()