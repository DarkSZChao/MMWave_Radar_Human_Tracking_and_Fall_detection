# Millimeter-Wave Radar-Based Multi-Human Tracking and Fall Detection System

## Overview
This repository contains the implementation and code resources for our paper: **Advanced Millimeter-Wave Radar System for Real-Time Multiple-Human Tracking and Fall Detection** (link: https://doi.org/10.3390/s24113660). The study explores an indoor system that employs Millimeter-Wave radars to track multiple humans and detect falls in real time. By integrating signals from non-intrusive radars, our framework addresses challenges such as mobility inconvenience, lighting conditions, and privacy issues inherent in wearable or camera-based systems.

### Key Features
- **Multi-Human Tracking**: Tracks multiple humans simultaneously with high precision.
- **Real-Time Fall Detection**: Accurately predicts and classifies human body statuses, including falls.
- **Advanced Signal Processing**: Employs Dynamic DBSCAN clustering and innovative feedback loops for enhanced accuracy.
- **Privacy and Accessibility**: Operates without cameras or wearables, ensuring non-intrusive monitoring. Camera module in the project is just for ground truth.

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
![System Flowchart Diagram](Sys_flowchart.jpg)


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

2. Install dependencies:
   pip install -r requirements.txt
