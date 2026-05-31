---
layout: home
permalink: index.html
repository-name: Signature-Verification-Using-ANN
title: Signature Verification Using Siamese Neural Networks
---

# Signature Verification Using Siamese Neural Networks

A deep learning project that achieves **97.92% test accuracy** on the CEDAR signature dataset by training a Siamese Neural Network to distinguish genuine signatures from forgeries.

---

## Team

- e20055, Dineth Shakya, [GitHub](https://github.com/DinethShakya23)

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Links](#links)

---

## Introduction

Handwritten signature verification is a biometric authentication problem with real-world applications in banking, legal, and identity verification workflows. Manual verification is slow and error-prone; automated systems require a model that can distinguish subtle differences between a genuine signature and a skilled forgery.

This project trains a **Siamese Neural Network** on the CEDAR signature dataset. Both signatures in a pair are passed through a shared CNN embedding model. The L1 absolute difference between their embeddings is then classified as genuine or forged by a small dense head.

Two comparison strategies were explored — concatenation of embeddings and L1 absolute difference — with the L1 approach achieving the best results (97.92% accuracy, 0.98 F1-score on both classes).

## Architecture

The shared embedding CNN extracts a feature vector from each 128×128 grayscale signature image through four convolutional blocks (64→128→256→512 filters). The two embeddings are compared via element-wise absolute difference, and a two-layer dense classifier produces a binary genuine/forged prediction.

## Results

| Metric | Forged | Genuine |
|---|---|---|
| Precision | 0.97 | 0.99 |
| Recall | 0.99 | 0.97 |
| F1-Score | 0.98 | 0.98 |
| **Test Accuracy** | | **97.92%** |

Training metrics and artifacts are tracked on [Weights & Biases](https://wandb.ai/e20055-university-of-peradeniya/Signature_Verification02?nw=nwusere20055).

## Links

- [Project Repository](https://github.com/DinethShakya23/Signature-Verification-Using-ANN){:target="_blank"}
- [W&B Dashboard](https://wandb.ai/e20055-university-of-peradeniya/Signature_Verification02?nw=nwusere20055){:target="_blank"}
- [CEDAR Dataset on Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset){:target="_blank"}
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
