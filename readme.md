# Air Quality PM2.5 Concentration High Innovative Prediction Model! A Parallel Prediction Model Based on VMD Rolling Decomposition + Attention Mechanism

Preface
With the acceleration of urban industrialization, air pollution issues, especially the prediction of PM2.5 concentrations, have become a focal point in environmental science and public health research. To address the non-stationarity and complexity of PM2.5 sequences, this paper proposes a parallel fusion prediction model based on Variational Mode Decomposition (VMD) for rolling decomposition and Transformer combined with a bidirectional Gated Recurrent Unit network (BiGRU) optimized by a global attention mechanism.
The method utilizes VMD to perform multimodal decomposition on time series signals, effectively isolating multiple frequency components within the signal. Subsequently, information from different components is synchronized and input into the multi-model architecture, fully exploiting the complex dynamic features of the sequence, thereby significantly enhancing prediction performance.
Dataset: Shanghai PM2.5 dataset
Environmental framework: Python 3.9, PyTorch 2.1 or above versions can be run
Use cases: Paper requirements, graduation design requirements
Code guarantee: Code comments are detailed and ready to run without modification.

1 Model Introduction
1.1 Overall Architecture Design
Transformer+BiGRU-Attention Parallel Neural Network Architecture:
Transformer module:
The Transformer module employs self-attention mechanisms to fully explore the global dependencies within sequences, benefiting from parallel computing and suitable for long sequences. This paper adopts multi-head attention mechanisms to capture multi-modal correlation features, enhancing the model's learning ability to complex fluctuation trends in air quality.
BiGRU-Attention module: A bidirectional GRU is an improved RNN capable of learning both forward and backward sequence information, suitable for capturing contextual sequential dependencies. This paper introduces a global attention mechanism to optimize the information conveyed by the hidden states, focusing on critical moments and features for prediction, thereby further enhancing representation quality and predictive performance.
1.2 Overall Process
Firstly, VMD is used to perform multi-scale decomposition on the original PM2.5 sequence, extracting different frequency components to enhance the regularity and predictability of the sequence; secondly, a parallel fusion architecture optimized with Transformer and GlobalAttention is designed to effectively capture both short-term and long-term dependencies as well as global features in the sequence. Experimental results show that the proposed model in this paper outperforms traditional models in terms of prediction accuracy and generalization ability, achieving high-precision predictions for changes in PM2.5 concentration levels, providing theoretical and methodological support for air quality forecasting.
1.3 Sliding Window and VMD Rolling Decomposition
(1) Data set import
This experiment uses the Shanghai PM2.5 dataset, totaling 35,142 samples with 17 features, focusing on the target variable data for visualization.
(2) VMD rolling decomposition
For multivariate time series data, after normalization, a sliding window technique is employed to divide the sequence into fixed-length sample windows, providing the foundation for VMD decomposition.
By sliding a fixed-size window over the original series, only the data within the window is processed. This means that new data is added after each iteration, and the oldest data in the window is removed. The number of subsequences generated remains constant through this process, preventing data leakage. Moreover, selecting the appropriate size for the sliding window is crucial. A small window leads to insufficient data, missing important information, and reduced prediction accuracy. Conversely, a large window increases computational complexity and running time, reducing available training data, and decreases model effectiveness.
2 Parallel Prediction Model Based on VMD+Transformer-BiGRU-Attention
\t\t2.1 Setting parameters and training the model
\tThe prediction effect of VMD+Transformer-BiGRU-Attention is significant. The VMD rolling decomposition enhances the ability to extract multi-scale features from complex time series, and by combining the parallel architecture of Transformer and BiGRU-Attention, it supplements global long-term and short-term dependencies. The model inputs multiple components decomposed by VMD into two subnetworks: Transformer and GlobalAttention-BiGRU, performs feature extraction, and then fuses them. This balances the advantages of both: Transformer captures global multi-scale dependencies; BiGRU focuses more on the local dynamics and temporal relationships in the sequence. Finally, the feature output is integrated through a fusion layer to enhance the stability and generalization capability of the model.
\t2.2 Visualization of prediction results
\t2.3 Model evaluation
Click to download: Complete data and Python code
Previously interesting content:
Time series prediction: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind speed prediction (I) Data introduction and preprocessing
Wind speed prediction (II) EMD-LSTM model based on Pytorch
Wind speed prediction (III) EMD-LSTM-Attention model
Wind speed prediction (IV) EMD-Transformer model based on Pytorch
Wind speed prediction (V) EMD-CNN-LSTM model based on Pytorch
Wind speed prediction (VI) EMD-CNN-GRU parallel model based on Pytorch
CEEMDAN + combined prediction model (BiLSTM-Attention + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-LSTM + ARIMA)
CEEMDAN + Combination Prediction Model (Transformer - BiLSTM + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-Transformer + ARIMA)
Multi-Feature Variable Sequence Forecasting (I) - CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (II) - CNN-LSTM-Attention Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (III) - CNN-Transformer Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (IV) Transformer-BiLSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (V) CEEMDAN+CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (VI) CEEMDAN+CNN-Transformer Wind Speed Forecasting Model
Predictive Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-BiLSTM-Attention
Predictive Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-Transformer-BiGRU
Superior Predictive Model: Secondary Decomposition-Combination Prediction
Multi-Feature Variable Sequence Forecasting (VII) CEEMDAN+Transformer-BiLSTM Predictive Model
Multi-Feature Variable Sequence Forecasting (VIII) Predictive Model Based on Particle Swarm Optimization Algorithm - CEEMDAN-SSA-BiLSTM
Multi-Feature Variable Sequence Forecasting (11) Predictive Model Based on Pytorch TCN-GRU
VMD + CEEMDAN Secondary Decomposition, CNN-LSTM Predictive Model
VMD + CEEMDAN Secondary Decomposition, CNN-Transformer Predictive Model
VMD + CEEMDAN 2nd Decomposition, Transformer-BiGRU Prediction Model
Prediction Model Based on the Plumage Optimization Algorithm SSA - Code All-in-One Kit
| Superior Combination Prediction Model!
High Innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention Prediction Model
| Superior Combination Prediction Model!
Lowest Price Online | Continuous Updates for the All-in-One Kit!
| Parallel Prediction Model Based on TCN-SENet + BiGRU-GlobalAttention
2nd Decomposition - Innovative Model Prediction All-in-One Kit
| BiTCN-BiGRU-CrossAttention Fusion of Spatiotemporal Features Highly Innovative Prediction Model
Update in Combination Predictions | Superior Combination Prediction Model!
Parallel Prediction Model with Fusion of Spatiotemporal Features by BiTCN-Transformer
| Based on Multi-level Attention Mechanism Parallel Prediction Model
| CEEMDAN-CNN-GRU-GlobalAttention + XGBoost Combination Prediction
Multi-step Prediction Series | LSTM, CNN, Transformer, TCN, Serial, Parallel Model Collection
END
\t2. Friends who like can click to follow, give the article a "like", share on social media or discuss groups

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
