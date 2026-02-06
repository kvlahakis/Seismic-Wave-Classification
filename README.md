# Seismic Phase Classification with RNNs and LSTMs

A Deep Learning project implementing Unidirectional and Bidirectional LSTM networks in PyTorch (benchmarked against standard RNNs) to classify seismic phases (P-waves and S-waves). This project evaluates how sequence modeling can improve the detection of geophysical events, with a focus on minority class recall.

## Project Overview
The detection of P-wave and S-wave arrivals is fundamental to seismology and geophysical exploration. This project utilizes a deep learning approach to automate this classification using sequential waveform data. By comparing different recurrent architectures, the project demonstrates the impact of temporal context on classification accuracy.

## Features
* **Custom Data Pipeline:** Implements a `SeqDataset` class to handle NumPy-based seismic data (`phaselink.npz`).
* **Architectural Comparison:** Implements and benchmarks three distinct models:
    * Standard Recurrent Neural Networks (RNN)
    * Unidirectional Long Short-Term Memory (LSTM)
    * Bidirectional Long Short-Term Memory (Bi-LSTM)
* **Training Framework:** A robust `Trainer` class featuring loss tracking, accuracy metrics, and precision/recall evaluation.
* **Visual Analysis:** Includes visualization of seismic signals (Time vs. Latitude) and model performance curves.

## Key Findings: The Power of Bidirectionality
The core finding of this analysis is the significant performance gain provided by the **Bidirectional LSTM** architecture. 

| Architecture | S-Wave Recall (Minority Class) |
| :--- | :--- |
| **Unidirectional LSTM (128 units)** | 0.7698 |
| **Bidirectional LSTM (128 units)** | **0.9008** |

### Why it works
The Bidirectional LSTM processes the seismic signal in both directions ($t_0 \rightarrow t_n$ and $t_n \rightarrow t_0$). This effectively allows the model to "see the future" of the waveform. In geophysics, the context following a wave onset is often just as critical as the preceding signal for accurate phase identification.



## Requirements
* Python 3.x
* PyTorch
* NumPy
* Matplotlib
* Scikit-learn

## Usage
1. Ensure `phaselink.npz` is in the project directory.
2. Run the Jupyter Notebook `seismic_classification.ipynb` to train the models and generate evaluation plots.

## Academic & Industrial Relevance
Modernizing geophysical exploration requires robust computational frameworks that can handle large-scale data with high precision. This project serves as a proof-of-concept for applying advanced numerical methods and sequence modeling to improve the accuracy of seismic data processing, a critical component in mineral and energy resource detection.

---
## License & Copyright
© 2026 Kieran Vlahakis
Licensed under the [MIT License](LICENSE)
