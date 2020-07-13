import numpy as np
import sys
from matplotlib import pyplot as plt


def main():

    train_number = int(sys.argv[1])
    test_number = int(sys.argv[2])
    pixel = int(sys.argv[3])
    shape_ratio = float(sys.argv[4])
    train_feature = sys.argv[5]
    train_label = sys.argv[6]
    test_feature = sys.argv[7]
    test_label = sys.argv[8]
    train_graph = sys.argv[9]
    test_graph = sys.argv[10]

    generate(train_number, pixel, shape_ratio, train_feature, train_label, train_graph)
    generate(test_number, pixel, shape_ratio, test_feature, test_label, test_graph)

    return


def sample_alpha(n):

    alphas = np.zeros(n)

    for c in range(n):
        alphas[c] = np.random.choice(range(1, 11)) * 0.1
    if len(np.unique(alphas)) != len(alphas):
        alphas = sample_alpha(n)

    return alphas


def draw_rectangle(canvas, alpha, fore, layer):

    limit = canvas.shape[0]

    lt_r = np.random.choice(range(1, int(limit / 3)))
    lt_c = np.random.choice(range(1, int(limit / 3)))
    rt_c = np.random.choice(range(int(limit / 3) * 2, int(limit)))
    ld_r = np.random.choice(range(int(limit / 3) * 2, int(limit)))

    canvas, layer = draw_element(canvas, layer, lt_r, lt_c, rt_c, ld_r, alpha, fore)

    return canvas, layer


def draw_angle(canvas, up, right, x, y, alpha, fore, layer):

    limit = canvas.shape[0]

    if up == 0:
        lt_r = np.random.choice(range(int(y + limit / 6), int(y + limit / 3)))
        lt_c = np.random.choice(range(int(x + 1), int(x + limit / 6)))
        rt_c = np.random.choice(range(int(x + limit / 3), int(x + limit / 2)))
        ld_r = np.random.choice(range(int(y + limit / 3), int(y + limit / 2)))
        canvas, layer = draw_element(canvas, layer, lt_r, lt_c, rt_c, ld_r, alpha, fore)

        ld_r = lt_r
        lt_r = np.random.choice(range(int(y), int(y + limit / 6)))

        if right == 0:
            rt_c = np.random.choice(range(int(x + limit / 6), int(x + limit / 3)))
        else:
            lt_c = np.random.choice(range(int(x + limit / 6), int(x + limit / 3)))
        canvas, layer = draw_element(canvas, layer, lt_r, lt_c, rt_c, ld_r, alpha, fore)
    else:
        lt_r = np.random.choice(range(int(y), int(y + limit / 6)))
        lt_c = np.random.choice(range(int(x), int(x + limit / 6)))
        rt_c = np.random.choice(range(int(x + limit / 3), int(x + limit / 2)))
        ld_r = np.random.choice(range(int(y + limit / 6), int(y + limit / 3)))
        canvas, layer = draw_element(canvas, layer, lt_r, lt_c, rt_c, ld_r, alpha, fore)

        lt_r = ld_r
        ld_r = np.random.choice(range(int(y + limit / 3), int(y + limit / 2)))

        if right == 0:
            rt_c = np.random.choice(range(int(x + limit / 6), int(x + limit / 3)))
        else:
            lt_c = np.random.choice(range(int(x + limit / 6), int(x + limit / 3)))

        canvas, layer = draw_element(canvas, layer, lt_r, lt_c, rt_c, ld_r, alpha, fore)

    return canvas, layer


def draw_element(canvas, layer, lt_r, lt_c, rt_c, ld_r, alpha, fore):

    canvas[lt_r:ld_r + 1, lt_c:rt_c + 1] = alpha

    if fore == 0:
        layer[lt_r:ld_r + 1, lt_c:rt_c + 1] = 0.2
    if fore == 1:
        layer[lt_r:ld_r + 1, lt_c:rt_c + 1] = 1.0

    return canvas, layer


def environment(number, pixel, shape_ratio):

    label = []
    feature = []
    combine = []

    for n in range(number):
        alphas = sample_alpha(2)
        canvas = np.zeros((pixel, pixel))
        layer = np.zeros(canvas.shape)

        for c in range(2):
            dice = np.random.rand()
            if dice < shape_ratio:
                canvas, layer = draw_rectangle(canvas, alphas[c], c, layer)
            else:
                x = np.random.choice([0, pixel / 2])
                y = np.random.choice([0, pixel / 2])
                canvas, layer = draw_angle(canvas, np.random.choice([0, 1]), np.random.choice([0, 1]), x, y, alphas[c],
                                           c, layer)
        label.append(layer)
        feature.append(canvas)
        combine.append(np.concatenate((canvas, layer), axis=1))
    f = np.reshape(feature, (-1, pixel))
    label = np.reshape(label, (-1, pixel))
    c = np.reshape(combine, (-1, pixel * 2))

    return f, label, c


def generate(number, pixel, shape_ratio, file_feature, file_label, graph):

    feature, label, combine = environment(number, pixel, shape_ratio)
    np.savetxt(file_feature, feature, fmt='%.1f')
    np.savetxt(file_label, label, fmt='%.1f')
    plt.imsave(graph, combine, cmap="gray")

    return


if __name__ == "__main__":
    main()
