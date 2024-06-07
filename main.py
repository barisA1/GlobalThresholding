import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Manually input the pixel intensity and count data
data = {
    'Intensity': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
    'Count': [12, 18, 32, 48, 52, 65, 55, 42, 32, 16, 10, 5, 18, 25, 32, 40, 65, 43, 32, 20, 10, 4]
}

df = pd.DataFrame(data)

# Convert the dataframe to a full list of pixel values
pixels = []
for index, row in df.iterrows():
    pixels.extend([row['Intensity']] * row['Count'])

pixels = np.array(pixels)

# Initialize the threshold
T0 = np.mean(pixels)

# Convergence threshold
threshold = 0.5

while True:
    # Divide into two groups
    G1 = pixels[pixels > T0]
    G2 = pixels[pixels <= T0]

    # Calculate mean values
    m1 = np.mean(G1) if len(G1) > 0 else 0
    m2 = np.mean(G2) if len(G2) > 0 else 0

    # New threshold
    T1 = (m1 + m2) / 2

    # Check for convergence
    if abs(T1 - T0) < threshold:
        break

    T0 = T1

# Print optimum threshold value
print(f"Optimum Threshold Value: {T0}")

# Calculate number of pixels above and below the threshold
G1_count = len(pixels[pixels > T0])
G2_count = len(pixels[pixels <= T0])

# Print results
print(f"Number of pixels above threshold: {G1_count}")
print(f"Number of pixels below or equal to threshold: {G2_count}")

# Plot the histogram and threshold
plt.hist(pixels, bins=range(100, 151), edgecolor='black')
plt.axvline(T0, color='red', linestyle='dashed', linewidth=1)
plt.title('Pixel Intensity Histogram with Threshold')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()


