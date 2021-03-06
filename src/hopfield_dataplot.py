# coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


image = np.array([[[-1., -1., 1., -1., -1.],
                   [-1., 1., -1., 1., -1.],
                   [-1., 1., -1., 1., -1.],
                   [-1., 1., -1., 1., -1.],
                   [-1., -1., 1., -1., -1.]],

                  [[-1., -1., 1., -1., -1.],
                   [-1., 1., 1., -1., -1.],
                   [-1., -1., 1., -1., -1.],
                   [-1., -1., 1., -1., -1.],
                   [-1., -1., 1., -1., -1.]],

                  [[-1., 1., 1., 1., -1.],
                   [1., -1., -1., 1., -1.],
                   [-1., -1., -1., 1., -1.],
                   [-1., -1., 1., -1., -1.],
                   [1., 1., 1., 1., 1.]],

                  [[-1., 1., 1., 1., -1.],
                   [-1., -1., -1., -1., 1.],
                   [-1, -1., 1., 1., 1.],
                   [-1., -1., -1., -1., 1.],
                   [-1., 1., 1., 1., -1.]],

                  [[-1., -1., 1., 1., -1.],
                   [-1., 1., -1., 1., -1.],
                   [1., 1., 1., 1., 1.],
                   [-1., -1., -1., 1., -1.],
                   [-1., -1., -1., 1., -1.]],

                  [[-1., 1., 1., 1., -1.],
                   [-1., 1., -1., -1., -1.],
                   [-1., 1., 1., 1., -1.],
                   [-1., -1., -1., 1., -1],
                   [-1., 1., 1., 1., -1.]]])

image2 = np.array([[[1., -1., -1., -1., -1.],
                    [1., -1., -1., -1., -1.],
                    [1., -1., -1., -1., -1.],
                    [1., -1., -1., -1., -1.],
                    [1., -1., -1., -1., -1.]],

                   [[-1., 1., -1., -1., -1.],
                    [-1., 1., -1., -1., -1.],
                    [-1., 1., -1., -1., -1.],
                    [-1., 1., -1., -1., -1.],
                    [-1., 1., -1., -1., -1.]],

                   [[-1., -1., 1., -1., -1.],
                    [-1., -1., 1., -1., -1.],
                    [-1., -1., 1., -1., -1.],
                    [-1., -1., 1., -1., -1.],
                    [-1., -1., 1., -1., -1.]],

                   [[-1., -1., -1., 1., -1.],
                    [-1., -1., -1., 1., -1.],
                    [-1., -1., -1., 1., -1.],
                    [-1., -1., -1., 1., -1.],
                    [-1., -1., -1., 1., -1.]],

                   [[-1., -1., -1., -1., 1.],
                    [-1., -1., -1., -1., 1.],
                    [-1., -1., -1., -1., 1.],
                    [-1., -1., -1., -1., 1.],
                    [-1., -1., -1., -1., 1.]]])

image3 = np.array([[[1., 1., 1., 1., 1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.]],

                   [[-1., -1., -1., -1., -1.],
                    [1., 1., 1., 1., 1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.]],

                   [[-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [1., 1., 1., 1., 1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.]],

                   [[-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [1., 1., 1., 1., 1.],
                    [-1., -1., -1., -1., -1.]],

                   [[-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1.],
                    [1., 1., 1., 1., 1.]]])


if __name__ == "__main__":
    num_pattern = 5

    fig, axs = plt.subplots(nrows=1, ncols=num_pattern, figsize=(3 * num_pattern, 5))

    for i in range(num_pattern):
        plot = (image3[i] + 1.) / 2.

        axs[i].imshow(plot, cmap=cm.Greys_r)

    fig.savefig("image3.png")
