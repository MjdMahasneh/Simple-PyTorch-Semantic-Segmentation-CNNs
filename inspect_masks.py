import cv2
import matplotlib.pyplot as plt


for i in range(0, 6):
    image_path = f'./val/images/image_{i}.jpg'
    mask_path = f'./val/masks/image_{i}.png'

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[0].set_title('image')
    ax[1].imshow(mask)
    ax[1].set_title('mask')
    plt.show()

