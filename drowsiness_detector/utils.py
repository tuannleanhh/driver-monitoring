import matplotlib.pyplot as plt


def show(image, label=None):
    plt.imshow(image)
    plt.xlabel(label)
    plt.show()
