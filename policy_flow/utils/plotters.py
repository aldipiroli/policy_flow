import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_image(image, path, cmap=None, dpi=300):
    h, w = image.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
