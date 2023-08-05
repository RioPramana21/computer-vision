import cv2
import matplotlib.pyplot as plt
from os import listdir

# 2440016804 - Rio Pramana - BA01

# Read target image
target_img = cv2.imread("./Dataset/Object.jpg")
target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# For the code not to be redundant, the object creation will be outside the loop
# Create SIFT object
sift = cv2.SIFT_create()
# Create FLANN object
index_params = dict(
    algorithm = 1,
    tree = 5
    # params for SIFT
)
search_params = dict(
    checks = 100
)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Feature detection of target image will also be outside the loop so its not redundant
# Get keypoints & descriptors from target image
target_kp, target_desc = sift.detectAndCompute(target_img_gray, None)

# Create plt.figure for comparison
plt.figure(1, figsize=(10, 8))

# Calculation for plotting
num_of_img = len(listdir("Dataset/Data"))
num_of_rows = int(len(listdir("Dataset/Data")) / 3)
if num_of_img % 3 != 0:
    num_of_rows += 1

# To save all results
all_result_img = []

# Read and process images from Data
for i, filename in enumerate(listdir("Dataset/Data")):
    # Read image
    data_img = cv2.imread("./Dataset/Data/" + filename)
    # Apply smoothing first before converting to grayscale
    # Preprocessing (Smoothing Gaussian)
    data_img_blur = cv2.GaussianBlur(data_img, (3, 3), 0)
    # Convert to grayscale
    data_img_gray = cv2.cvtColor(data_img_blur, cv2.COLOR_BGR2GRAY)
    # Get keypoints & descriptors from data image
    data_kp, data_desc = sift.detectAndCompute(data_img_gray, None)
    # Get matches
    matches = flann.knnMatch(target_desc, data_desc, k = 2)
    # Filter matches using Lowe's ratio test
    mask = [[0, 0] for i in range(len(matches))]
    for j, (m, n) in enumerate(matches):
        if (m.distance < n.distance * 0.7):
            mask[j] = [1, 0]
    # Draw matches
    result = cv2.drawMatchesKnn(
        target_img, target_kp,
        data_img, data_kp,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        matchesMask=mask
    )
    # Plot matches to be shown
    # Convert to RGB first because we are using matplotlib
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    all_result_img.append(result)

    plt.subplot(num_of_rows, 3, i + 1)
    plt.imshow(result)
    plt.title(filename + " Matches Result")
    plt.xticks([])
    plt.yticks([])

# Show comparison of all matches
plt.show()

# Show the best match result
# Based on the comparison, the best match result is with the file "5 - Chiki Balls.jpg"
plt.figure("Buy the Chiki", figsize=(10, 8))
plt.imshow(all_result_img[5])
plt.title("Best Match Result")
plt.xticks([])
plt.yticks([])
plt.show()