import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


img = cv2.imread('./park.jpg', cv2.IMREAD_UNCHANGED)
if img is None:
    raise Exception("Couldn't load the image")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, n_channel = img.shape
img_features = img.reshape((-1, n_channel)).copy()
out_features = img_features.copy()
img_features = img_features.astype(float)
n_pixels = h * w
big_window_size = 30
small_window_size = 15
tolerance = 5
chosen_before_idx = np.zeros((n_pixels,), dtype=bool)
chosen_before_fixed = np.arange(n_pixels)
counter = 0
t = time.time()

while True:
    counter += 1
    not_chosen = np.argwhere(chosen_before_fixed > 0)
    print(len(not_chosen))
    # After first iterations set the window size smaller
    if counter > 5:
        big_window_size = 20
        small_window_size = 5
        tolerance = 10
    # Ignore the last 100000 remaining pixels
    if len(not_chosen) < 100000:
        break
    # Choose a random point from not chosen points
    n = np.random.randint(0, len(not_chosen))
    # Save pixels corresponding to this segment in this variable
    chosen_now_idx = np.zeros((n_pixels,), dtype=bool)
    feature_chosen = img_features[not_chosen[n], :].copy()
    # print(feature_chosen)
    while True:
        # Distance from center
        distance = np.sqrt(np.sum(np.square(img_features - feature_chosen), axis=1))
        # print(distance)
        # print(distance[n])
        # Label points in the trajectory
        chosen_now_idx[distance < small_window_size] = True
        # Big window
        big_window_condition = distance < big_window_size
        big_window = img_features[big_window_condition, :].copy()
        feature_chosen_prev = feature_chosen.copy()
        # Calculate mean of the cluster
        feature_chosen = np.mean(big_window, axis=0)
        # Calculate the mean shift
        mean_shift = feature_chosen - feature_chosen_prev
        mean_shift_norm = np.sqrt(np.sum(np.square(mean_shift)))
        print(mean_shift_norm)
        # Convergence criterion
        if mean_shift_norm < tolerance:
            break
    # Also label the vectors in big window around high density area
    distance = np.sqrt(np.sum(np.square(img_features - feature_chosen), axis=1))
    # Update visited pixels
    chosen_now_idx = np.logical_or(chosen_now_idx, distance < big_window_size)
    segment_mean = np.mean(img_features[chosen_now_idx, :], axis=0)
    out_features[np.logical_and(chosen_now_idx, ~chosen_before_idx), :] = segment_mean
    delete_idx = np.logical_and(chosen_now_idx, ~chosen_before_idx)
    chosen_before_fixed[delete_idx] = -1
    # print(chosen_before_fixed.min())
    chosen_before_idx = np.logical_or(chosen_before_idx, chosen_now_idx)
    # if counter % 20 == 0:
    #     temp = img_features.reshape((h, w, 3))
    #     plt.imshow(temp)
    #     plt.show()
    # if counter % 10 == 0:
    #     temp = out_features.reshape((h, w, 3))
    #     try:
    #         plt.imsave('test2.jpg', temp)
    #     except:
    #         pass

print(time.time() - t)
temp = out_features.reshape((h, w, 3))
temp = cv2.medianBlur(temp, 7)
plt.imshow(temp)
plt.imsave('res05.jpg', temp)
