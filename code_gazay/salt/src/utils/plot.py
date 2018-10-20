import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from . import TestBatch
plt.style.use('seaborn-white')
sns.set_style("white")


# Fancy edges from mask/pred(with threshold) finding
def edges(mask, threshold):
    mask = mask > threshold
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    return edges


def show_images(dataset, unprocess_fn, model=None, max_images=30, threshold=0):
    grid_width = 10
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width * 2, grid_height * 2))
    title = "Green: salt"
    if model:
        title = title + '; Red: pred'
    for i, idx in enumerate(dataset.train[:max_images]):
        img = dataset.image(idx)
        mask = dataset.mask(idx)
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.imshow(edges(mask, threshold), alpha=0.4, cmap="Greens")
        if model:
            batch = TestBatch()._image(dataset, img)
            pred = model(batch).data[0].cpu().numpy()[0]
            pred = unprocess_fn(pred)
            ax.imshow(pred, alpha=0.2, cmap="OrRd")
            ax.imshow(edges(pred, threshold), alpha=0.3, cmap="OrRd")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle(title)
