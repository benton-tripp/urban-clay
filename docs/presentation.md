# Slide 1: Title Slide
## Utilizing the Clay Foundation Model and Sentinel-2 Imagery for Urban Growth Monitoring in Johnston County, NC
### Benton Tripp

**Dialogue**:
Good [morning/afternoon], everyone. Thank you for joining my presentation. Today, I’ll discuss the application of the Clay Foundation Model and Sentinel-2 imagery for monitoring urban growth in Johnston County, North Carolina. This research addresses the challenge of infrequent updates in urban imperviousness datasets by integrating multispectral satellite imagery and advanced machine learning techniques.

---

# Slide 2: Research Motivation
## Why Monitor Urban Growth?
- Urbanization impacts ecosystems, water cycles, and local climates.
- Existing datasets (e.g., NLCD) update approximately every 5 years.
- Need for frequent monitoring to guide sustainable urban planning.

**Dialogue**:
Urban growth fundamentally alters ecosystems and local climates, making it crucial to monitor impervious surface expansion like roads and buildings. While datasets like the National Land Cover Database (NLCD) provide valuable insights, their updates every five years are insufficient for capturing rapid urbanization. This creates a need for more frequent and precise monitoring.

---

# Slide 3: Study Area
## Johnston County, North Carolina
- Location: Eastern NC, part of the Raleigh metropolitan area.
- Rapid urban expansion over recent decades.
- Study area divided into 600×600-meter tiles.

![Placeholder for Map of Johnston County](file-tfYeZbEhAtlCCAmBkiEB7TjE)

**Dialogue**:
Johnston County, part of the rapidly growing Raleigh metropolitan area, was selected for its ongoing urban expansion. The study divides the county into 600×600-meter tiles, ensuring manageable spatial units for analysis and modeling.

---

# Slide 4: Data Overview
## Key Datasets Used
| **Dataset**        | **Source**        | **Key Attributes**                |
|---------------------|-------------------|------------------------------------|
| Sentinel-2 Imagery | ESA               | 10m resolution, 13 spectral bands |
| NLCD Imperviousness | USGS              | 30m resolution, updated every 5 years |
| County Boundary    | Johnston GIS      | Vector boundary data              |

**Dialogue**:
The study leverages three main datasets: Sentinel-2 imagery from the European Space Agency, which provides multispectral data at a high spatial resolution; urban imperviousness data from the NLCD; and official boundary data from Johnston County GIS. These datasets form the foundation for the analysis.

---

# Slide 5: Sentinel-2 Imagery Example
## April 2016, Johnston County
![Sentinel-2 Imagery Example Placeholder](file-UU515Cpa760yXcuOtpwpvKAO)

**Dialogue**:
Here is an example of Sentinel-2 imagery for Johnston County from April 2016. These images, processed for minimal cloud cover, provide key spectral information across multiple bands, enabling detailed urban growth analysis.

---

# Slide 6: Research Objectives
## Goals
- Leverage the Clay Foundation Model for feature extraction.
- Predict urban imperviousness percentages at high spatial and temporal resolution.
- Develop optimized neural network models using hyperparameter tuning.

**Dialogue**:
The primary goal is to utilize the Clay Foundation Model to extract spectral embeddings from Sentinel-2 imagery. These embeddings are then used to predict urban imperviousness percentages. Neural networks, optimized through hyperparameter tuning, are employed to improve predictive accuracy.

---

# Slide 7: Data Preparation
## Workflow Overview
1. Divide study area into 600×600-meter tiles.
2. Extract spectral features using Clay Foundation Model.
3. Resample imperviousness data to match tile resolution (200m).
4. Normalize features and targets for model compatibility.

**Dialogue**:
The data preparation involves dividing the county into 600×600-meter tiles and extracting features using the Clay Foundation Model. Urban imperviousness data is resampled to 200-meter resolution, striking a balance between spatial detail and computational efficiency.

---

# Slide 8: Spatial Clustering for Data Splits
## KMeans Clustering
- 30 clusters based on geographic coordinates.
- 70% training, 10% validation, 20% testing.
- Reduces spatial autocorrelation in evaluation.

![Cluster Visualization Placeholder](file-pl6eolPRKtbHW4Om4jiJ86eX)

**Dialogue**:
To ensure representative sampling, tiles were clustered geographically using KMeans. This reduces spatial autocorrelation and ensures that training, validation, and testing subsets reflect the study area's diversity. A 70:10:20 ratio was applied to allocate clusters to these subsets.

---

# Slide 9: Neural Network Architectures
## Models Used
| Model        | Description                       |
|--------------|-----------------------------------|
| **SNN**      | Simple Neural Network (1 hidden layer) |
| **DNN**      | Deep Neural Network (2 hidden layers) |

- Both optimized through hyperparameter tuning.
- Regularization techniques: dropout, weight decay.

**Dialogue**:
Two neural network architectures were implemented: a simple neural network with one hidden layer and a deeper variant with two hidden layers. Both models were fine-tuned through hyperparameter optimization, employing dropout and weight decay to prevent overfitting.

---

# Slide 10: Methods Recap
## Key Steps
1. Prepare data: Tile grids, feature extraction, and normalization.
2. Split data: KMeans clustering and subset allocation.
3. Train models: Optimize SNN and DNN architectures.
4. Evaluate: Use MSE and MAE metrics on the test set.

**Dialogue**:
To recap the methods, the study prepared the data by creating tile grids and extracting features. KMeans clustering was used for spatial splits, and both SNN and DNN architectures were optimized for training. Finally, evaluation metrics like MSE and MAE were applied to assess performance.

---

