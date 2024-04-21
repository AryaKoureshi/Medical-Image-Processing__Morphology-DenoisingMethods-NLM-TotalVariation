"""
Medical-Image-Processing__Morphology-DenoisingMethods-NLM-TotalVariation
Arya Koureshi(aryakoureshi.github.io)
arya.koureshi@gmail.com
"""

# %%
import cv2
import matplotlib.pyplot as plt

# %% part a
A = cv2.imread('C:/Users/aryak/OneDrive/Desktop/MAM/HW02_MIAP_Corrected/HW02_MIAP_Corrected/Theorem/image.png', cv2.IMREAD_GRAYSCALE)
_, A = cv2.threshold(A, 120, 1, cv2.THRESH_BINARY)

B = A[656: 691, 651:685]

C = cv2.erode(A, B, iterations=1)

plt.figure()
plt.suptitle('Erosion')
plt.subplot(1, 3, 1)
plt.imshow(A, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('A')
plt.subplot(1, 13, 7)
plt.imshow(B, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('B')
plt.subplot(1, 3, 3)
plt.imshow(C, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('C')
plt.tight_layout()
plt.show()

# %% part b
D = cv2.dilate(C, B, iterations=1)

plt.figure()
plt.suptitle('Dilation')

plt.subplot(1, 3, 1)
plt.imshow(C, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('C')
plt.subplot(1, 13, 7)
plt.imshow(B, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('B')
plt.subplot(1, 3, 3)
plt.imshow(D, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('D')
plt.tight_layout()
plt.show()

# %% part c
E = cv2.dilate(D, B, iterations=1)

plt.figure()
plt.suptitle('Dilation')
plt.subplot(1, 3, 1)
plt.imshow(D, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('D')
plt.subplot(1, 13, 7)
plt.imshow(B, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('B')
plt.subplot(1, 3, 3)
plt.imshow(E, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('E')
plt.tight_layout()
plt.show()

# %% part d
F = cv2.erode(E, B, iterations=1)
plt.figure()
plt.suptitle('Erosion')
plt.subplot(1, 3, 1)
plt.imshow(E, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('E')
plt.subplot(1, 13, 7)
plt.imshow(B, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('B')
plt.subplot(1, 3, 3)
plt.imshow(F, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('F')
plt.tight_layout()
plt.show()
