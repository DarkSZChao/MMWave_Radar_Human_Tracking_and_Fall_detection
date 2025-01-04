# Millimeter-Wave Radar-Based Multi-Human Tracking and Fall Detection System

## Overview
This repository contains the implementation and resources for our **Millimeter-Wave Radar-Based Indoor Tracking and Fall Detection System**, leveraging **three Millimeter-Wave radars from Texas Instruments**. Our system is designed to non-intrusively track multiple humans and detect falls in real time, addressing challenges such as mobility inconvenience, lighting conditions, and privacy issues inherent in wearable or camera-based systems.

### Key Features
- **Multi-Human Tracking**: Tracks up to three humans simultaneously with high precision.
- **Real-Time Fall Detection**: Accurately predicts and classifies human body statuses, including falls.
- **Advanced Signal Processing**: Employs Dynamic DBSCAN clustering and innovative feedback loops for enhanced accuracy.
- **Privacy and Accessibility**: Operates without cameras or wearables, ensuring non-intrusive monitoring.

---

## Abstract
This study explores an indoor system for tracking multiple humans and detecting falls, employing **Millimeter-Wave radars**. Our framework integrates signals from multiple radars to track positions and predict body statuses in real time. Key contributions include:
- Evaluation of radar characteristics, including resolution, interference, and coverage.
- Introduction of Dynamic DBSCAN clustering based on signal energy levels.
- Development of a probability matrix for target tracking and fall detection.
- Implementation of a feedback loop for noise reduction.
  
Through extensive evaluation (300+ minutes, ~360,000 frames), our prototype system achieves:
- **Precision**: 98.9% for single-target tracking, 96.5% for two targets, 94.0% for three targets.
- **Fall Detection Accuracy**: 96.3%.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Evaluation](#evaluation)
5. [Contributions](#contributions)
6. [License](#license)

---

## System Architecture
### Components
- **Radar Hardware**: Three **Millimeter-Wave radars** from Texas Instruments.
- **Signal Processing Pipeline**:
  - Dynamic DBSCAN clustering
  - Probability matrix for target tracking
  - Feedback loop for noise reduction
- **Real-Time Framework**: Integrates radar signals to track and classify human activity.

### Workflow
1. **Signal Acquisition**: Raw data collected from the radars.
2. **Clustering and Tracking**: Data processed using clustering algorithms and tracking matrices.
3. **Fall Detection**: Status prediction using advanced probability-based methods.
4. **Output**: Visual and statistical tracking results.

---

## Installation
### Prerequisites
- Python 3.8 or higher
- Libraries: `numpy`, `scipy`, `matplotlib`, `pandas`, `sklearn`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mmwave-human-tracking.git
   cd mmwave-human-tracking
