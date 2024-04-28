import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img_path = 'hw2_img.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Calculate the histogram
hist_before = cv2.calcHist([img], [0], None, [256], [0, 256])

def histogram_equalization(image):
    # Compute the cumulative distribution function (cdf)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Find the minimum nonzero value of the cdf
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Map the image pixels to equalized value
    img_equalized = cdf[image]

    return img_equalized

# Equalize the image histogram
img_eq = histogram_equalization(img)

# Calculate the histogram of the equalized image
hist_after = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# Save the equalized image
output_path = 'histogram_equalized_image.png'
cv2.imwrite(output_path, img_eq)

# Plot the histograms and the images
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Original image and histogram
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[1, 0].plot(hist_before)
axs[1, 0].set_title('Histogram Before Equalization')

# Equalized image and histogram
axs[0, 1].imshow(img_eq, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[0, 1].axis('off')

axs[1, 1].plot(hist_after)
axs[1, 1].set_title('Histogram After Equalization')

fig2, axs2 = plt.subplots(1, 2, figsize=(10, 7))

axs2[0].plot(hist_before)
axs2[0].set_title('Histogram Before Equalization')

axs2[1].plot(hist_after)
axs2[1].set_title('Histogram After Equalization')

plt.savefig('histogram_equalized_apply_before_after.png')

# Show the plots
plt.tight_layout()
plt.show()