# Medical-Image-Processing__Morphology-DenoisingMethods-NLM-TotalVariation

This repository contains Python code for various tasks related to medical image analysis and processing. Below are the detailed descriptions and implementations for each section.

## Section 1: Morphology

The goal of this section is to create a mask for the man in the picture and process it accordingly.

- **Step 1:** Determine the range of clothing pixels.
  - Read the image `q1.png`.
  - Identify clothing pixels based on the green channel.
  - Save the resulting mask as `q1res01.jpg`.
- **Step 2:** Cover image holes and create a uniform mask for clothes.
  - Use closing and opening methods to fill holes and smooth the mask.
  - Save the uniform mask as `q1res02.jpg`.
- **Step 3:** Redden all areas belonging to the person's clothes.
  - Utilize the obtained mask to identify clothes area.
  - Redden the clothes area in the original image and save as `q1res03.jpg`.

## Section 2: Denoising Methods

This section focuses on denoising techniques applied to medical images.

- **Step 1:** Gaussian noise addition and grayscale conversion.
  - Add Gaussian noise to the image `hand.jpg`.
  - Convert the color image to grayscale.
  - Display noisy and clean images side by side.
- **Step 2:** Classical Regression Filtering.
  - Explain Gaussian kernel formulation and its role in denoising.
  - Remove noise using the Gaussian kernel.
- **Step 3:** Bilateral Filtering.
  - Explain Bilateral Filtering and its parameters.
  - Remove noise using Bilateral Filtering.
  - Discuss the conceptual meaning of parameters `hx` and `hg`.

## Section 3: NLM

This section involves noise removal using the Non-Local Means (NLM) algorithm and Gaussian filter.

- **Step 1:** Apply noise to images and clean using NLM.
  - Add Gaussian and pepper-salt noises to images.
  - Clean the images using the NLM algorithm with varying parameters.
  - Plot PSNR graphs for different filter parameters.
- **Step 2:** Compare NLM with Gaussian filter.
  - Denoise images using Gaussian filter with different parameters.
  - Calculate PSNR between denoised and original images.
  - Compare results and discuss findings.

## Section 4: Total Variation

This section implements the total variation filter for noise reduction.

- **Step 1:** Create a modified Shepp-Logan phantom with Gaussian noise.
  - Generate clean and noisy images.
- **Step 2:** Remove noise using total variation filter.
  - Implement the total variation filter with specified parameters.
  - Display clean images, noisy images, and noise removal results.
- **Step 3:** Calculate SNR criterion for the filter.

## Section 5: Total Variation Reproduction

This section reproduces the denoising operation using total variation.

- **Step 1:** Implement the provided codes for TV denoising.
- **Step 2:** Apply denoising operation on the noisy image.
- **Step 3:** Analyze and report results according to the provided description.

For detailed implementation and results, refer to the respective Python files in this repository.
