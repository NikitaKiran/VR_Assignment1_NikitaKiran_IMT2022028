import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    image_blur_gray = cv2.GaussianBlur(img_grey, (5,5), 0)
    return image_blur_gray

def edge_and_coin_detection(image_blur_gray, image):
    # Canny edge detection
    canny = cv2.Canny(image_blur_gray, 90, 255)
    plt.imshow(canny, cmap='gray')
    plt.axis('off')
    plt.title("Canny Edge Detection")
    plt.savefig('coin_detection_outputs/coin-detection-canny.jpg')

    # Close the edges using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Find contours
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw detected contours
    contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 6)

    # Display detected contours
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.imshow(contour_image)
    plt.title("Detected Coins")
    plt.savefig('coin_detection_outputs/coin-detection-contours.jpg')
    return contours

def segment_and_count(contours, image):
    min_coin_area = 5000
    coins = 0
    # Create a mask
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Segmenting coins
    seg_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f'coin_detection_outputs/segmented_coins.jpg', seg_image)

    individual_coins = []
    for i, contour in enumerate(contours):
        # Skip small contours
        if cv2.contourArea(contour) < min_coin_area:
            continue

        # Create a mask for the current coin
        mask = np.zeros_like(image[:, :, 0])
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)


        # Extract the coin from the original image
        coin = cv2.bitwise_and(image, image, mask=mask)

        # Get bounding box around the coin
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 

        # Crop the coin and add to the list
        individual_coins.append(coin[y:y+h, x:x+w])

        coins += 1

    _, axes = plt.subplots(1, len(individual_coins), figsize=(15, 5))
    for ax, coin in zip(axes, individual_coins):
        ax.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    plt.suptitle(f"Detected {coins} Coins", fontsize=30)

    plt.savefig('coin_detection_outputs/individual_coins.jpg')
    return coins

# Load the image
image = cv2.imread('inputs/coin_detection_input.jpg')

# Preprocess the image
image_blur_gray = preprocess_image(image)

# Detect edges and contours
contours = edge_and_coin_detection(image_blur_gray, image)

# Segment and count the coins
final_count = segment_and_count(contours, image)
print(f"Total number of coins detected: {final_count}")