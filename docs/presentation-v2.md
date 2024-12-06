---

I am about 80% of the way done with this paper/analysis, and need to make a presentation on the work so far. The presentation should be 10+ minutes in length, and include the following:

1. Introduction/Background: problem, motivation for the research, research question / objective
2. Study site: where, why this site, geographic characteristics
3. Data: sources, resolution, spatial extent, processing and integration issues
4. Methods: describe methods for analysis and modeling, why you selected them and what is your general workflow
5. Results and discussion: present and explain the results
    - qualitative and quantitative, tables, graphs, maps/images
    - discuss impact of data and methods on the results, uncertainty, accuracy,
    - compare with results from other studies – confirms previously observed phenomena, shows something new,
      which questions remain unresolved 
    - what still needs to be done for the paper
6. Conclusion: summary of the most important findings including advances in methodology, future work

Below is what I have put together so far (For each slide I have also included the Dialogue that I will read out loud when recording the presentation):

---

# Utilizing the Clay Foundation Model and Sentinel-2 Imagery for Urban Growth Monitoring in Johnston County, NC

Benton Tripp

Dialogue: Hello, my name is Benton Tripp. Today I will present a proof-of-concept study that integrates Sentinel-2 multispectral satellite imagery with the Clay Foundation Model, an open-source earth foundation model for Earth observation, to monitor urban growth in Johnston County, North Carolina. The central motivation arises from the need for more frequent and timely estimates of urban growth than what standard land cover products, such as the National Land Cover Database (NLCD), currently provide. By leveraging foundation models and deep learning, this goal of this study is to address these temporal gaps, thereby informing more sustainable urban planning and environmental management.

Beyond improving the update frequency and accuracy of urban imperviousness mapping, this research also illustrates how emerging “foundation models” for Earth observation can fill some of the voids in existing data and workflows. By harnessing these models’ advanced representation learning capabilities, it becomes possible to quickly adapt to a variety of remote sensing tasks, scaling analyses with minimal additional training data. This approach demonstrates the broader potential of such models to enhance current geospatial monitoring efforts, ultimately facilitating more responsive and data-driven decision-making.

---

## Introduction

* Rapid urbanization alters land use and ecology
* Impervious surfaces track urban expansion 
* Five-year NLCD data updates often miss rapid changes
* Timely imperviousness estimates support better planning

(Figure) NLCD 2016 Urban Imperviousness - Johnston County, NC

Dialogue: Urbanization reshapes landscapes, alters ecosystems, and influences resource distribution. Impervious surfaces—roads, buildings, parking lots—function as key indicators of urban expansion, influencing hydrological processes, local climates, and ecological integrity. Although the National Land Cover Database (NLCD) provides authoritative baseline data on impervious surfaces, a five-year update cycle leaves gaps in capturing rapid transformations. The NLCD’s imperviousness estimates, developed from high-resolution training data and applied to Landsat imagery using machine learning, integrate roads, buildings, and other ancillary datasets to yield nuanced insights. Yet static temporal resolutions fail to represent evolving urban landscapes adequately.

To address these deficiencies, this analysis integrates Sentinel-2 satellite imagery with the Clay Foundation Model, a recently developed Earth observation foundation model. The combined approach produces more frequent and detailed urban imperviousness estimates, aligning the NLCD’s rich impervious surface products with a higher-temporal-resolution method. This demonstration underscores how emerging Earth observation foundation models can supplement existing datasets, enhancing the timeliness and responsiveness of urban growth assessments without relying on extensive new training data.

## Motivation

* Rapid development within Johnston County, NC could be benefited by more granular monitoring
* Five-day revisit frequency from Sentinel-2 supports high-temporal-resolution analysis
* Foundation models allow more data-efficient adaptation to evolving remote sensing tasks

(Figure) Coverage and revisit time for Sentinel-2 MSI acquisitions

Dialogue: Johnston County’s status as a rapidly growing area within the Raleigh-Durham-Chapel Hill metropolitan region identifies this location as an ideal case study. The Sentinel-2 five-day revisit cycle provides frequent, high-resolution imagery suitable for detecting changes at finer temporal scales. The emergence of foundation models in Earth observation, analogous to large-language models in natural language processing, offers a scalable approach. Pre-trained on extensive datasets, models such as the Clay Foundation Model can be adapted efficiently to downstream tasks with minimal additional labeling, thereby reducing data acquisition burdens associated with traditional supervised methodologies.

## Research Question and Objectives

Question: Does integrating a foundation model with Sentinel-2 data improve temporal resolution and accuracy of urban imperviousness mapping?

Objective: Establish a scalable, data-efficient framework that generates more frequent urban imperviousness estimates than the five-year NLCD cycle.

Dialogue: The primary research question addresses whether integration of the Clay Foundation Model and Sentinel-2 data can enhance temporal resolution and accuracy in urban imperviousness mapping. The objective involves establishing a scalable, data-efficient framework to produce imperviousness estimates more frequently than the standard five-year update interval of datasets such as NLCD. By using pre-trained foundation model embeddings, optimized neural network architectures, and hyperparameter tuning, the proposed approach delivers timely insights into evolving urban conditions, providing a basis for adaptive planning and sustainable resource allocation.

## Study Site

* Johnston County, North Carolina  
* Area: ~2,050 km²  
* Rapidly developing within Raleigh-Durham-Chapel Hill region
* EPSG:32617

(Figure) Johnston County

Dialogue: Johnston County is located in eastern North Carolina between latitudes 35.3°N and 35.8°N, and longitudes 78.0°W and 78.6°W. Its proximity to a major metropolitan region makes it a prime location to observe and model urban expansion. By using a local Universal Transverse Mercator (UTM) projection (EPSG:32617), spatial accuracy is improved by minimizing distortions common in geographic coordinate systems. This area provides a realistic and relevant environment to test the framework’s ability to capture the nuances of rapidly changing urban landscapes.

## Data Sources and Extent

* Sentinel-2 MSI imagery: 10m resolution, RGB & NIR bands (out of 13 available spectral bands)
* NLCD imperviousness data: 30m resolution, updated every ~5 years  
* STAC APIs for data access (Microsoft Planetary Computer, AWS Earth Search)

(Figure) Map displaying the boundary of Johnston County, North Carolina, overlaid with a 600x600-meter tile grid.

Dialogue: The primary data sources include Sentinel-2 multispectral images accessed via SpatioTemporal Asset Catalog (STAC)-compliant APIs, which define a standardized structure for describing geospatial assets and thus ensure accessible, interoperable, and reproducible workflows. National Land Cover Database urban imperviousness data serve as the ground truth, albeit at a slightly coarser temporal resolution of 30 meters compared to the 10 meter resolution of the Sentinel-2 images. To manage computational loads and establish a clear framework for modeling, the county was divided into 600×600-meter tiles. This division creates uniform spatial units and simplifies the alignment of spectral and imperviousness data. As a result, the geospatial information is transformed into a structured, grid-based format that can be more readily incorporated into the machine learning workflows used in this project.

## Data Processing Steps

* Reproject boundaries to EPSG:32617  
* Generate 600×600m grid tiles  
* Resample urban imperviousness raster (to 200m), clip to 600x600m tiles
* Align Sentinel-2 images with tiles

(Figure) Sentinel-2 imagery of Johnston County in April 2016, overlaid with a 600x600-meter tile grid used and the county boundary

Dialogue: When processing the data, first, county boundaries were reprojected to EPSG:32617 and arranged into a uniform grid of 600×600-meter tiles. Urban imperviousness data were resampled from 30m to 200m for computational efficiency when developing the predictive models, producing a 3×3 matrix per tile. Sentinel-2 imagery was clipped and normalized to these tiles, focusing on key spectral bands (Blue, Green, Red, NIR) to preserve essential spectral information. Seasonal and annual variation were incorporated by selecting a total of 20 images from 2016 through 2024, with each image chosen to maintain minimal cloud cover.

## Temporal Data Selection

* Sentinel-2: ~3 images per year (2016-2024)  
* Filtered by <1% cloud cover  
* Balanced temporal coverage

Dialogue: Ensuring balanced temporal coverage required selecting approximately three cloud-free Sentinel-2 images per year. The chosen distribution captured seasonal variations and reduced temporal clustering. Although certain dates were unavailable due to weather or data gaps, careful filtering preserved a broad range of environmental conditions, enhancing the robustness of training and validation.

## Earth Foundation Model Embeddings

* Clay Model: pretrained on large Earth Observation datasets  
* Extract spatial embeddings 
* No manual feature engineering needed

Dialogue: The Clay Foundation Model is an open-source Earth observation foundation model designed to capture a wide range of environmental conditions and patterns from satellite imagery. This model leverages a Vision Transformer architecture optimized for geospatial and temporal data, enabling it to understand not only static spectral information but also the context of location and seasonality. Unlike traditional approaches that demand extensive manual feature engineering, this model applies self-supervised learning methods—specifically, a Masked Autoencoder technique—to learn robust representations directly from the input data. As a result, the model produces embeddings that efficiently condense complex environmental information into a manageable numeric form.In this study, the Clay model was applied as a pretrained backbone to extract 768-dimensional spatial embeddings from each tile’s Sentinel-2 imagery. These embeddings are already aligned with patterns commonly found in Earth observation data, reducing the dependence on manually curated features or external indices. Augmenting these core embeddings, additional temporal and spatial information was incorporated through sine/cosine transformations of dates and normalized latitude/longitude values. This integrated feature set provides a comprehensive input for modeling urban imperviousness, capturing not only spectral distinctions but also the temporal dynamics and geographic context essential for more accurate and timely predictions.

## Earth Foundation Model Embedding Flow

Spectral-Spatial Data (pixels):

- 4 bands (Blue, Green, Red, NIR)
    * Each band normalized by known mean and standard deviation
- Spatial resolution: 10 m
- Tile size: 600 m × 600 m
- Resulting pixel matrix: 60 × 60
- (Overall spectral-spatial input: 4 × 60 × 60)

Temporal Embeddings:

- Hour of day (h), normalized as:
    * sin(h*2π/24), cos(h*2π/24)
- Week of year (w), normalized as:
    * sin(w*2π/52), cos(w*2π/52)

Spatial Embeddings:

- Tile centroid latitude (lat) and longitude (lon) in radians
- Normalized as: 
    * sin(lat), cos(lat)
    * sin(lon), cos(lon)

Ground Sample Distance (GSD):

- 10 meters

** Sine/cosine transformations ensure cyclical patterns are properly represented

(Figure depicting flow - see description below)

(1) Some sample Sentinel-2 Images, representing the raw data
(2) Datacube for each Date/Tile:
    - Spectral-Spatial Tensor: (4 × 60 × 60)
    - Temporal Vector: 4 scalars
    - Spatial Vector: 4 scalars
    - Waves Vector: 4 scalars
    - GSD: 1 scalar
(3) Clay foundation model
(4) For each Date/Tile:
    - Vector of embedded data: (768 x 1) 

Dialogue: The embedding process begins by integrating raw Sentinel-2 imagery and contextual information into a unified datacube. The spectral-spatial portion comprises four multispectral bands (Blue, Green, Red, NIR) sampled at 10-meter resolution, forming a 4 × 60 × 60 pixel matrix that represents a 600-meter by 600-meter tile. Each band is normalized by known mean and standard deviation values to ensure consistency across varying scenes. Temporal information, including the hour of day and week of year, undergoes sine and cosine transformations to capture cyclical patterns. The tile’s centroid latitude and longitude are similarly transformed to produce stable spatial references. Ground sample distance and per-band wavelength data are also included to reflect sensor and scene characteristics.

After assembling these components—spectral-spatial data, temporal vectors, spatial embeddings, wavelength metadata, and GSD—into the datacube, the Clay Foundation Model converts this rich, multidimensional input into a 768-dimensional embedding. By compressing extensive raw data into a lower-dimensional feature vector, the model preserves essential information while enabling more efficient downstream analysis. This approach reduces the need for manual feature engineering and better positions predictive models, such as those for estimating urban imperviousness, to leverage the intricate interplay of spectral patterns, seasonal variations, and geographic context.

## Target Variable: Urban Imperviousness

* NLCD imperviousness as ground truth  
* Resampled to 200m (3×3 matrix per tile)  
* Normalized values [0,1]

Each 3×3 target matrix is paired with a 768-dimensional feature vector generated by the Clay Foundation Model, integrating Sentinel-2 imagery with temporal and spatial context at the corresponding tile.

(Figure) Urban imperviousness across Johnston County in 2016 overlayed with 600x600m tile grid

Dialogue: Once the embedding process is established, the next component is the target variable representing urban imperviousness. Using the National Land Cover Database (NLCD) imperviousness layer as ground truth, values are resampled from their native 30-meter resolution to 200-meter cells. This resampling translates each 600×600-meter tile into a 3×3 matrix of imperviousness percentages. Normalizing these percentages to a [0,1] range ensures numerical stability and compatibility with the modeling framework.

As part of this project, obtaining and preparing the urban imperviousness data incorporated several concepts from the GIS582 course. Various resampling methods were explored, including nearest-neighbor and cubic interpolation, but the differences in data variation were minimal. Bilinear interpolation was ultimately selected as it provided a balance between computational efficiency and the preservation of continuous data characteristics when downsampling from 30 to 200 meters.

Each of these 3×3 target matrices is then paired with the 768-dimensional feature vector generated by the Clay Foundation Model. By linking the imperviousness targets directly to the embeddings—which already encapsulate spectral, temporal, and spatial information—subsequent models can learn relationships between observed urban development patterns and the underlying landscape signals. This alignment sets the stage for more accurate and timely estimates of urban growth.

## Data Splitting and Validation

- Training, validation, and testing restricted to 2016
- Sentinel-2 imagery provides multiple seasonal dates per tile
- Urban imperviousness assumed constant in 2016 with minimal intra-annual variation
- Spatial stratification ensures geographic consistency
- Post-2016 data reserved for visualization and exploration

(Figure) Stratified splits of 600x600m tiles for training (blue), validation (green), and testing (red) sets

Dialogue: The training, validation, and testing datasets were restricted to 2016, consistent with the availability of National Land Cover Database imperviousness data. Sentinel-2 imagery, however, includes multiple acquisition dates within the year, allowing the model to incorporate seasonal spectral variability. While urban imperviousness is assumed to remain constant throughout 2016, slight changes within the year may occur due to temporary or seasonal variations in land use or surface cover. This assumption introduces a small margin of error, but it was necessary to ensure a cohesive training framework.

Data splitting was conducted spatially using KMeans clustering on tile centroids. All dates for a given tile were retained within the same split, ensuring geographic consistency and avoiding spatial leakage. Testing and validation were also limited to 2016, ensuring that performance metrics reflect valid comparisons with the ground truth. Post-2016 data, while excluded from model training and testing, is used for visualization and exploratory analysis, demonstrating the framework’s applicability to future datasets.

## Neural Network Architectures

1. Simple Neural Network (SNN):
Input (768) → Hidden (128) → Output (9)

2. Deep Neural Network (DNN):
Input (768) → Hidden (128,128) → Output (9)

Dialogue: TODO: two-paragraph, high-level explanation of neural networks and deep learning

... *** Then adjust the dialogue below to flow from the explanation. Make sure to avoid pronouns!

Two feed-forward neural network architectures were tested. The SNN served as a baseline neural approach with a single hidden layer. The DNN introduced an additional hidden layer, enabling the model to learn more complex mappings. Both used ReLU activations in the hidden layers and sigmoid activations at the output to ensure predicted imperviousness values remain between zero and one. We optimized hyperparameters such as learning rate, weight decay, and dropout rates to enhance generalization.

## Model Training Details

* Loss function: Mean Squared Error (MSE)
* Regularization techniques: Dropout, Weight Decay, Early Stopping
* Baseline: Mean imperviousness from training data

(Figure) Training and validation loss curves for the DNN model (128 hidden units, 10% dropout, 0.0001 weight decay, 0.0005 learning rate, 10-epoch patience, MSE loss)

Dialogue: Model training prioritized minimizing the mean squared error (MSE) between the predicted and observed urban imperviousness values. The MSE loss function was selected to emphasize large prediction errors more heavily, providing a robust metric for guiding the optimization process. This focus ensures that the models learn to capture subtle but meaningful variations in urban imperviousness rather than simply approximating average values.

Regularization techniques, such as dropout and weight decay, were implemented to address the challenge of overfitting, particularly given the high-dimensional feature space generated by the Clay-derived embeddings. Dropout selectively disables neurons during training, preventing the network from becoming overly reliant on specific features, while weight decay imposes a penalty on large model weights to promote simpler and more generalizable solutions. Additionally, early stopping was employed to terminate training once the validation loss plateaued, preventing unnecessary iterations that could lead to overfitting.

A baseline model, defined as the mean imperviousness value across the training data, served as a reference point for evaluating the neural network architectures. The baseline approach captures overall trends in the data but lacks the capacity to account for spatial and temporal nuances. Demonstrating consistent improvements over this baseline underscores the ability of the neural networks to extract meaningful relationships from the spectral, temporal, and spatial features encoded in the embeddings.

## Preliminary Results

* DNN outperforms baseline and SNN
* Predicted imperviousness aligns with known urban areas  
* Some seasonal and data gaps remain

(Figure) Comparison of RMSE for the top 125 SNN and DNN models, with the baseline RMSE as a reference (red dashed line).

Dialogue: Initial results demonstrate that the deep neural network (DNN) architectures tested tended to surpasses both the simple neural network (SNN) architectures and the baseline model in performance, as evidenced by lower model metrics. These results indicate that the DNN effectively leverages the Clay Foundation Model embeddings to translate spectral and contextual features into accurate urban imperviousness estimates. Predictions align closely with known urbanized areas in Johnston County, capturing spatial patterns at a 600×600-meter scale.

Despite these promising results, some uncertainties remain due to seasonal variability in Sentinel-2 imagery and occasional data gaps. Seasonal changes in vegetation or surface reflectance may introduce slight errors, particularly in areas where spectral features shift across seasons. Addressing these challenges through further temporal validation and enhanced preprocessing might help with improving the robustness and reliability of the model.

## Predicting Post-2016 Data

* Evaluate generalization on unseen data
* Predictions align with observed urban growth trends
* Detect changes in imperviousness from 2016 to 2024

The predictive framework was applied to post-2016 data to evaluate its ability to generalize to unseen temporal intervals. Initial predictions align with observed trends in urban growth, capturing expansions in impervious surfaces consistent with known development patterns in Johnston County. This capability demonstrates the framework’s potential for detecting dynamic changes in urban imperviousness over time. Future steps include validating these predictions against additional ground truth data to further assess accuracy.

## Next Steps in this Project

- Residual analysis to identify patterns in prediction errors
- Quantify uncertainty in predictions

Dialogue: Although the model development and training phases of this project are complete, the current focus is on analyzing the results to better understand the model’s performance and limitations. This includes conducting a detailed residual analysis to uncover patterns in the prediction errors, which may reveal specific areas where the model under- or overestimates urban imperviousness. Additionally, quantifying uncertainty in the predictions at the tile level will provide insights into the confidence of the model across different spatial and temporal regions.

## Future Work

- Fine-tune embeddings for regional specificity
- Explore convolutional architectures for spatial feature learning
- Incorporate additional spectral indices or auxiliary data
- Train on multiple years for a more robust model
- Extend analysis to other geographic areas

Future work might focus on fine-tuning the Clay Foundation Model embeddings to better capture region-specific patterns of urban growth. Investigating convolutional architectures may provide enhanced spatial feature learning, improving the model’s ability to represent localized urbanization trends. Incorporating auxiliary data, such as vegetation indices, nighttime lights, or socioeconomic variables could further improve model performance. Extending this methodology to other regions experiencing rapid urbanization would validate the framework’s scalability and robustness.

## Summary and Conclusions

* Foundation models provide scalable solutions for urban monitoring
* Sentinel-2 imagery enables high-temporal-resolution analysis
* The framework demonstrates feasibility for frequent imperviousness mapping
* Future enhancements will focus on scalability and accuracy

In summary, this study highlights the potential of integrating foundation models with Sentinel-2 imagery for urban growth monitoring, as well as similar remote-sensing applications. The proposed framework leverages the high temporal resolution of Sentinel-2 and the advanced feature representation capabilities of the Clay Foundation Model to generate frequent and reliable imperviousness estimates. These findings demonstrate a scalable and data-efficient approach to urban monitoring, addressing the limitations of traditional datasets. Future efforts will aim to enhance model accuracy, extend analysis to additional regions, and validate the framework’s applicability in broader contexts.

---