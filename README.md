# Traffic Congestion Prediction with ASTGCN

## 1. Project Overview  
Traffic congestion prediction is a critical challenge in modern smart city planning and transportation management. By forecasting future traffic speeds or flow across a road network, city authorities can manage traffic signals, route planning, and congestion mitigation in real time.  
In this project, we apply the Attention-based Spatio-Temporal Graph Convolutional Network (ASTGCN) to predict traffic speeds on a road graph, leveraging both spatial and temporal dependencies in the data.

## 2. Key Features  
- **Graph representation of the road network**: We model the road network as a graph, where nodes represent intersections or road segments, edges represent connectivity or traffic links.  
- **Multi-temporal dependencies**: The model incorporates three temporal views — recent (hour-level), daily (day-lag), and weekly (week-lag) sequences.  
- **Attention mechanisms**: Spatial attention captures dynamic inter-node relations, temporal attention captures time-step dependencies.  
- **Chebyshev Graph Convolutions**: We use Chebyshev polynomials to efficiently perform graph convolution over the adjacency matrix.  
- **Unified pipeline**: Data ingestion (JSON files for edges, nodes, traffic, weather) → canonical dataset → adapter for ASTGCN input → model training → evaluation.  
- **TensorBoard logging & model checkpointing**: Loss curves tracked during training; best model automatically saved based on validation performance.

## 3. Project Structure  
```plaintext
astgcn_v1/
├── traffic_astgcn/
│   ├── data/
│   │   ├── canonical.py
│   │   ├── adapters/
│   │   │   └── astgcn_adapter.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   └── astgcn.py
│   │
│   ├── training/
│   │   ├── train_astgcn.py
│   │   ├── evaluate_best.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── data/
│   ├── raw/
│   │   └── runs/
│   │       ├── run_.../
│   │       │   ├── nodes.json
│   │       │   ├── edges.json
│   │       │   ├── traffic_edges.json
│   │       │   └── weather_snapshot.json
│   │       └── ...
│   │
│   └── processed/
│       └── canonical.parquet
└── astgcn_best.pth
└── README.md
```

## 4. Results  
After training the model, we achieved the following performance on the test set using the best checkpoint:

- MAE ≈ 0.546  
- RMSE ≈ 1.022  
- MAPE ≈ 0.026  
- R² ≈ 0.977  

These results indicate that the model performs very well in capturing the spatio-temporal dynamics of traffic and generalizes well on unseen data.

## 5. Reference  
Wen, W., et al. (2019). *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting*. AAAI. https://doi.org/10.1609/aaai.v33i01.3301922  
