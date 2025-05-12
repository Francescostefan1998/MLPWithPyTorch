import numpy as np
def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    print(f'w_rot: {w_rot}')
    print(f'x_padded: {x_padded}')
    if p > 0:
        zero_pad = np.zeros(shape=p)
        print(f'zero_pad: {zero_pad}')
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
        print(f'x_padded: {x_padded}')
    res = []
    for i in range(0, int((len(x_padded) - len(w_rot))) + 1, s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
    return np.array(res)

# Testing
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]
print('Conv1d Implementation:', conv1d(x, w, p=2, s=1))
print('NumPy Results:', np.convolve(x, w, mode='same'))