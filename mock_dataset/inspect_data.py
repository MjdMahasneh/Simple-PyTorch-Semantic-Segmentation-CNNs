import cv2
import matplotlib.pyplot as plt

for i in range(0, 6):
    image_path = f'./val/images/image_{i}.jpg'
    mask_path = f'./val/masks/image_{i}.png'

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale

    # Show image and mask side by side
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    ax[0].set_title('Image')

    # Display mask with a colormap
    ax[1].imshow(mask, cmap='jet', interpolation='nearest')  # 'jet' colormap or choose any other
    ax[1].set_title('Mask')
    plt.show()

