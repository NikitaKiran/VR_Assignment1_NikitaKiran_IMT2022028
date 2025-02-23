import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_image(title, img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'panorama_outputs1/{title}.jpg')

def detect_keypoints_and_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k = 2)
    # Apply Lowe's ratio test
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return matches

def view_matches(image1, image2, keypoints1, keypoints2, matches, i):
    matches.sort(key=lambda x: x.distance)
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_image(f"keypoint_matches_{i-1}_and_{i}", img_matches)

def homography_matrix(keypoints1, keypoints2, matches):
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warp_images(panorama, image, H, i):
    h1, w1 = panorama.shape[:2]
    h2, w2 = image.shape[:2]
    new_width = w1 + w2
    new_height = max(h1, h2)
        
    # Warping the images to align with panorama
    warped_img = cv2.warpPerspective(image, H, (new_width, new_height))
    # save_image(f"warped_{i}", warped_img)
    extended_panorama = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # Place the existing panorama on the left
    extended_panorama[:h1, :w1] = panorama
    # save_image(f"extended_panorama_{i}", extended_panorama)
    mask1 = (extended_panorama > 0).astype(np.float32)
    mask2 = (warped_img > 0).astype(np.float32)

    # Blending the images(linear blending)
    blended = (extended_panorama * mask1 + warped_img * mask2) / (mask1 + mask2 + 1e-8)  # Prevent division by zero
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    # save_image(f"blended_{i}", blended)
    return blended
    

input_images = ["inputs/panorama_inputs1/1.jpg", "inputs/panorama_inputs1/2.jpg", "inputs/panorama_inputs1/3.jpg"]
# input_images = ["inputs/panorama_inputs/IMG_6495.png", "inputs/panorama_inputs/IMG_6497.png", "inputs/panorama_inputs/IMG_6499.png", "inputs/panorama_inputs/IMG_6501.png"]
images = [cv2.imread(image) for image in input_images]
panorama = images[0]
for i in range(1, len(images)):
    keypoints1, descriptors1 = detect_keypoints_and_descriptors(panorama)
    if i == 1:
        imagecopy = panorama.copy()
        cv2.drawKeypoints(imagecopy, keypoints1, imagecopy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        save_image("keypoints1", imagecopy)
    keypoints2, descriptors2 = detect_keypoints_and_descriptors(images[i])
    imagecopy = images[i].copy()
    cv2.drawKeypoints(imagecopy, keypoints2, imagecopy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_image(f"keypoints{i+1}", imagecopy)
    matches = match_keypoints(descriptors1, descriptors2)
    view_matches(images[i - 1], images[i], keypoints1, keypoints2, matches, i)
    H = homography_matrix(keypoints1, keypoints2, matches)
    panorama = warp_images(panorama, images[i], H, i)
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    panorama = panorama[y:y+h, x:x+w]

gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(thresh)
x, y, w, h = cv2.boundingRect(coords)
panorama = panorama[y:y+h, x:x+w]
save_image("final-panorama", panorama)
    
