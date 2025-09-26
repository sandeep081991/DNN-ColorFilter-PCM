# Broadband-Tunable Multispectral Imaging Using Phase-Change-Based Linear Variable Filters

This repository contains code and models for simulating **multispectral imaging** using a **phase-change material (PCM) based Linear Variable Filter (LVCF)**. The goal is to demonstrate how tunable spectral filtering, combined with a **deep neural network (DNN) surrogate model**, enables **broadband, compact, and efficient multispectral imaging systems**.

## What We Did

* **Designed PCM-based LVCF** with tunable transmission characteristics across a broad spectral range.
* **Developed a DNN surrogate model** to predict filter transmission quickly, replacing heavy full-wave simulations.
* **Simulated multispectral imaging** by applying the predicted filter response to synthetic scenes.
* Demonstrated how the proposed approach can enable **broadband, real-time, and compact multispectral cameras** without bulky filter wheels.

## Features

* End-to-end simulation pipeline:

  1. Load trained DNN surrogate model.
  2. Generate synthetic multispectral scene.
  3. Apply PCM-LVCF transmission for tunable imaging.
* Modular design: extendable for new materials, wavelengths, and imaging tasks.

## Applications

* Remote sensing
* Biomedical imaging
* Compact surveillance cameras
* Astronomy and catadioptric system integration

## Requirements

* Python 3.9+
* TensorFlow / Keras
* NumPy, Matplotlib, scikit-learn

## How to Run

```bash
python multispectraliamging.py
```

This will:

1. Load the pretrained DNN model.
2. Generate a synthetic hyperspectral scene.
3. Simulate LVCF-filtered outputs across wavelengths.

