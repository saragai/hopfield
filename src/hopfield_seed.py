# coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt


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


def print_image(_im):
    for _i in range(_im.shape[0]):
        row = ""
        for _j in range(_im.shape[1]):
            row += ". " if _im[_i, _j] == -1. else "X "
        print(row)
    print("----------")


def train(_data):
    _q = _data.shape[0]
    _w = np.zeros([_data.shape[1], _data.shape[1]])
    for _iq in range(_q):
        xiq = _data[_iq, :]
        _w += xiq @ xiq.T
    _w /= _q
    for _d in range(_data.shape[1]):
        _w[_d, _d] = 0.
    return _w


def update(_w, _x, _threshold=0.):
    _i = np.random.randint(0, _x.size)
    _x[_i, 0] = 1. if (_w @ _x)[_i, 0] >= _threshold else -1.
    return _x


def randomize(x, rate):
    _x = np.copy(x)
    mask = np.random.random_sample(_x.shape) <= rate
    noise = np.random.random_sample(_x.shape)
    noise[noise > 0.5] = 1.
    noise[noise <= 0.5] = -1.

    _x[mask] = noise[mask]
    return _x


if __name__ == "__main__":

    num_pattern = 6

    print("==== train ====")
    data = image.reshape([num_pattern, -1, 1])

    # print(w)

    iter_num = 100

    fig, axs = plt.subplots(nrows=2, ncols=num_pattern, figsize=(3 * num_pattern, 5))
    # noise_rates = np.arange(21)/100
    noise_rates = np.array([0., 0.05, 0.1, 0.15, 0.2])

    color_array = ['r', 'b', 'y', 'g', 'c', 'm']

    _data = data[:num_pattern]
    w = train(_data)

    # num of pattern
    for s in range(100):
        np.random.seed(s)

        for qi in range(num_pattern):
            print(f"  No.{qi}")
            x = _data[qi]
            accs = np.zeros(len(noise_rates))  # Accuracy rates
            sims = np.zeros(len(noise_rates))  # Degrees of similarity

            # noise
            for ni, noise_rate in enumerate(noise_rates):
                acc = 0
                sim = 0
                acc_flip = 0
                test_x = None
                for i in range(iter_num):
                    test_x = randomize(x, noise_rate)
                    # print_image(test_x.reshape([5, 5]))
                    for _ in range(100):
                        test_x = update(w, test_x, 0)
                        # print_image(test_x.reshape([5, 5]))

                    if np.allclose(test_x, x):
                        acc += 1
                    sim += np.linalg.norm(test_x - x)

                # print_image(test_x.reshape([5, 5]))
                print(f"noise rage({noise_rate}) : {acc / iter_num}")
                accs[ni] = acc / iter_num
                sims[ni] = sim / iter_num

            if s == 99:
                axs[0, qi].plot(noise_rates, accs, color=color_array[qi], alpha=0.01, label=f"image {qi}")
            else:
                axs[0, qi].plot(noise_rates, accs, color=color_array[qi], alpha=0.01)
            axs[1, qi].plot(noise_rates, sims, color=color_array[qi], alpha=0.01)

    for qi in range(num_pattern):
        axs[0, qi].set_title(f"{qi} Image's Accuracy rate")
        axs[1, qi].set_title(f"{qi} Image's Similarity")
        # axs[0, q].set_xlabel("noise rate")
        axs[1, qi].set_xlabel("noise rate")
        axs[0, qi].set_ylim(0., 1.)
        axs[1, qi].set_ylim(0., 7.)

    axs[0, 0].set_ylabel("accuracy")
    axs[1, 0].set_ylabel("similarity")
    # fig.legend()

    fig.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.4)
    fig.savefig(f"hopfield1.png")
