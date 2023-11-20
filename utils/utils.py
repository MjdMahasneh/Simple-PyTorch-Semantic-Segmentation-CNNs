from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt



# def plot_losses(train_losses, val_losses, save=True, save_dir=None):
#
#     if save:
#         assert save_dir is not None, 'Please provide a valid save directory'
#         save_dir = Path(save_dir + 'losses.png')
#
#     ## plot the training and validation losses
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('iterations')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Training and Validation Loss')
#     plt.savefig(save_dir)  # Save the figure
#     plt.show()



import matplotlib.pyplot as plt
import os








def write_to_file(file_path, content):
    with open(file_path, 'a') as f:
        f.write(content)
    return





def plot_training_progress(iter_ids, train_losses, iter_overall_iou_scores, iter_per_class_iou_scores, show=True, save=False, save_dir=None, id=None):

    if save:
        assert save_dir is not None, 'Please provide a valid save directory'

    plt.figure(figsize=(10, 6))

    # Scaling train losses to the range [0, 1] for better visualization/convenience
    scaled_train_losses = (train_losses - np.min(train_losses)) / (np.max(train_losses) - np.min(train_losses))

    # Plotting scaled training loss
    plt.plot(range(len(scaled_train_losses)), scaled_train_losses, label='Train Loss (scaled)', color='red', linewidth=0.1)

    # Plotting overall IoU scores
    plt.plot(iter_ids, iter_overall_iou_scores, label='Overall IoU Score', color='black', linewidth=1.0)

    # Plotting per-class IoU scores
    for i, scores in iter_per_class_iou_scores.items():
        plt.plot(iter_ids, scores, label=f'Class {i} IoU Score', marker='o', linestyle='dashed', linewidth=1.0, markersize=3)

    plt.xlabel('Iteration')
    plt.ylabel('Scores/Loss (Scaled)')
    plt.title('Training Loss and Scores over Iterations')
    plt.legend()
    plt.grid(True)



    if save:
        ## make the directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot
        if id is not None:
            plot_path = os.path.join(save_dir, f"training_progress_plot_{id}.png")
        else:
            plot_path = os.path.join(save_dir, "training_progress_plot.png")

        plt.savefig(plot_path)

    if show:
        plt.show()
    else:
        # Close the plot to free up memory
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

    for c in range(1, 3):  # for each channel (R, G, B)
        for label, color in zip(unique_labels, colors):
            img_with_mask[:, :, c] = np.where(mask == label,
                                              img_with_mask[:, :, c] * (1 - alpha) + alpha * color[c],
                                              img_with_mask[:, :, c])

    return img_with_mask







# if __name__ == '__main__':
#
#     # Mock data
#     num_iters = 100
#     iter_ids = list(range(0, num_iters, 10))  # Iterations where IoU scores are recorded
#     num_classes = 3  # Example for 3 classes
#
#     train_losses = np.random.uniform(0.1, 0.5, num_iters)  # Mock training loss values
#     iter_overall_iou_scores = np.random.uniform(0.5, 0.8, len(iter_ids))  # Mock overall IoU scores
#     iter_per_class_iou_scores = {i: np.random.uniform(0.4, 0.7, len(iter_ids)) for i in
#                                  range(num_classes)}  # Mock per-class IoU scores
#
#     # Call the function
#     plot_training_progress(iter_ids, train_losses, iter_overall_iou_scores, iter_per_class_iou_scores, show=True, save=True, save_dir='./temp')
#






