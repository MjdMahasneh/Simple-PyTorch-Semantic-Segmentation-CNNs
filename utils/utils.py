from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os


def write_to_file(file_path, content):
    with open(file_path, 'a') as f:
        f.write(content)
    return


def plot_training_progress(iter_ids, train_losses, iter_overall_iou_scores, iter_per_class_iou_scores, show=True,
                           save=False, save_dir=None, id=None):
    if save:
        assert save_dir is not None, 'Please provide a valid save directory'

    plt.figure(figsize=(10, 6))

    scaled_train_losses = (train_losses - np.min(train_losses)) / (np.max(train_losses) - np.min(train_losses))

    plt.plot(range(len(scaled_train_losses)), scaled_train_losses, label='Train Loss (scaled)', color='red',
             linewidth=0.1)

    plt.plot(iter_ids, iter_overall_iou_scores, label='Overall IoU Score', color='black', linewidth=1.0)

    for i, scores in iter_per_class_iou_scores.items():
        plt.plot(iter_ids, scores, label=f'Class {i} IoU Score', marker='o', linestyle='dashed', linewidth=1.0,
                 markersize=3)

    plt.xlabel('Iteration')
    plt.ylabel('Scores/Loss (Scaled)')
    plt.title('Training Loss and Scores over Iterations')
    plt.legend()
    plt.grid(True)

    if save:

        os.makedirs(save_dir, exist_ok=True)

        if id is not None:
            plot_path = os.path.join(save_dir, f"training_progress_plot_{id}.png")
        else:
            plot_path = os.path.join(save_dir, "training_progress_plot.png")

        plt.savefig(plot_path)

    if show:
        plt.show()
    else:

        plt.close()

    return


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def overlay_mask_on_image(image, mask, colors=None, alpha=0.5):
    img_with_mask = image.copy()

    unique_labels = np.unique(mask)

    if colors is None:
        colors = [tuple(np.random.randint(0, 255, 3)) for _ in unique_labels]

    for c in range(1, 3):
        for label, color in zip(unique_labels, colors):
            img_with_mask[:, :, c] = np.where(mask == label,
                                              img_with_mask[:, :, c] * (1 - alpha) + alpha * color[c],
                                              img_with_mask[:, :, c])

    return img_with_mask
