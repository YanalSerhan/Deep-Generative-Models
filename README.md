# Deep Generative Models

This repository contains a collection of Jupyter notebooks exploring major families of deep generative models. Each notebook focuses on a different approach to modeling complex data distributions, with an emphasis on understanding the core ideas, mathematical foundations, and practical implementation patterns behind modern generative modeling.

The project is organized as a compact study set for learning how deep learning models can generate images, approximate complex distributions, and model uncertainty in latent spaces.

## Repository Structure

- `VAE.ipynb` — Variational Autoencoder notebook
- `Normalizing Flows.ipynb` — Normalizing flow notebook
- `Autoregressive Image Modeling.ipynb` — autoregressive image generation notebook
- `Denoising Diffusion Probabilistic Models.ipynb` — diffusion model notebook

## Overview of the Models Included

This repository covers four foundational deep generative modeling frameworks:

1. Variational Autoencoders (VAEs)
2. Normalizing Flows
3. Autoregressive Models
4. Denoising Diffusion Probabilistic Models (DDPMs)

These models are among the most important techniques in modern generative AI and are commonly used for image generation, density estimation, representation learning, and sampling from high-dimensional data.

---

## File-by-File Documentation

### VAE.ipynb

This notebook introduces the Variational Autoencoder (VAE), one of the most widely used latent-variable generative models.

Core idea:
A VAE learns a compressed latent representation of data by encoding inputs into a latent distribution and then reconstructing the original input from that latent space. Instead of learning a single fixed latent vector, the model learns a probability distribution in latent space, which allows it to sample new data points.

Key concepts covered:
- Encoder-decoder architecture
- Latent variable modeling
- Reparameterization trick
- Mean and variance of latent variables
- Reconstruction loss
- KL divergence regularization
- Evidence lower bound (ELBO)

Why it matters:
VAEs are foundational in generative modeling because they combine probabilistic inference with deep learning. They provide a principled way to model uncertainty and create smooth latent spaces that can be sampled to generate new examples.

What to expect in this notebook:
- Data preprocessing and encoding
- Neural network implementation of encoder and decoder
- Training objective for reconstruction + KL penalty
- Latent space visualization and sampling
- Example generation of new samples

Use cases:
- Image generation
- Latent representation learning
- Feature disentanglement and interpolation
- Anomaly detection

---

### Normalizing Flows.ipynb

This notebook covers normalizing flows, a class of generative models that learn complex probability distributions by transforming a simple base distribution through a sequence of invertible mappings.

Core idea:
A normalizing flow starts from a simple distribution such as a Gaussian and applies a series of invertible transformations. Because each transformation is invertible and has a tractable Jacobian, the final model can compute exact likelihoods and generate samples.

Key concepts covered:
- Change of variables
- Invertible transformations
- Jacobian determinant
- Flow layers and coupling layers
- Exact density estimation
- Sampling from transformed distributions

Why it matters:
Unlike VAEs, which optimize a lower bound, normalizing flows can learn exact likelihoods. They are especially useful when we need to estimate how likely a data point is under the model and when we want to sample from a learned distribution while preserving tractable probability calculations.

What to expect in this notebook:
- Basic probability transformations
- Implementation of flow-based layers
- Training with log-likelihood objective
- Evaluation of density estimation quality
- Sampling synthetic outputs

Use cases:
- Density estimation
- High-likelihood generative modeling
- Probabilistic inference
- Structured data generation

---

### Autoregressive Image Modeling.ipynb

This notebook focuses on autoregressive models for image generation, where data is generated one element at a time conditioned on previous elements.

Core idea:
Autoregressive models model the joint distribution of pixels as a product of conditional distributions. For an image, pixels are generated sequentially, usually in row-major order, with each new pixel conditioned on the pixels already generated.

Key concepts covered:
- Sequential data generation
- Conditional probability factorization
- Pixel-by-pixel modeling
- Masked convolutions
- Raster scan ordering
- Likelihood-based training

Why it matters:
Autoregressive models are one of the earliest and still highly effective approaches for structured generation, especially in images and sequence data. They are conceptually simple and produce tractable likelihoods, though generation can be slower than more modern diffusion-based methods.

What to expect in this notebook:
- Modeling image data as a sequence of pixel values
- Conditional generation of image patches or pixels
- Use of masks to preserve causality
- Training with negative log-likelihood
- Sampling images one step at a time

Use cases:
- Image generation
- Compression and modeling of spatial dependencies
- Structured sequence modeling
- Generative modeling of discrete data

---

### Denoising Diffusion Probabilistic Models.ipynb

This notebook introduces Denoising Diffusion Probabilistic Models (DDPMs), which are currently among the most powerful and widely used generative models for image synthesis.

Core idea:
Diffusion models define a forward process that gradually adds Gaussian noise to data over many steps, and a reverse process that learns to remove this noise step by step. By learning to denoise corrupted versions of the data, the model can generate realistic samples from noise.

Key concepts covered:
- Forward diffusion process
- Reverse denoising process
- Noise schedules
- U-Net architecture
- Training on noisy inputs
- Sampling from learned reverse process
- Latent denoising dynamics

Why it matters:
Diffusion models often produce higher-quality samples than many earlier generative approaches and are especially strong in image generation. They are now a standard benchmark in modern generative modeling and a central area of research in AI.

What to expect in this notebook:
- Forward noising process and timestep scheduling
- Machine learning model trained to predict noise
- Reverse denoising trajectory
- Image generation from random noise
- Visualization of sample quality and training behavior

Use cases:
- High-quality image synthesis
- Text-to-image generation
- Data augmentation
- Creative generation and editing

---

## Suggested Learning Order

To understand the progression of generative modeling ideas, the notebooks are best explored in the following order:

1. `VAE.ipynb`
   - Learn latent variable modeling and probabilistic representation learning

2. `Normalizing Flows.ipynb`
   - Learn exact likelihood modeling through invertible transformations

3. `Autoregressive Image Modeling.ipynb`
   - Learn sequential generation and conditional modeling

4. `Denoising Diffusion Probabilistic Models.ipynb`
   - Learn modern state-of-the-art generation through denoising

This order moves from foundational latent-variable methods to exact-likelihood modeling, then to sequential generation, and finally to modern diffusion-based generation.

---

## Why This Repository Is Useful

This repository is valuable for anyone learning generative modeling because it collects several classic and modern approaches in a single, compact project. It is especially useful for:

- Students studying deep learning and generative AI
- Practitioners wanting a practical introduction to generative methods
- Researchers comparing different paradigms
- Learners who prefer notebook-based experimentation over pure theoretical reading

The notebooks are designed to help users connect mathematical concepts with actual implementations and visual outputs.

---

## How to Use

1. Open the repository in a Jupyter environment or notebook viewer.
2. Start with the notebook that matches the model family you want to study.
3. Read the explanations and run the code cells in order.
4. Experiment with training parameters, architecture changes, and sample generation behavior.

---

## Summary

This repository is a curated set of educational notebooks covering key deep generative modeling methods. It includes:

- VAEs for latent-space learning
- Normalizing flows for exact likelihood modeling
- Autoregressive models for sequential image generation
- Diffusion models for state-of-the-art generative sampling

Together, these notebooks provide a strong conceptual and practical foundation in modern generative modeling.
