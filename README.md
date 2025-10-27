# interview-question--phd
1. Data and Preprocessing

Q1. How did you handle missing values in your dataset?
A1. I checked for missing values and imputed them using median imputation, which is robust to outliers, ensuring the integrity of the dataset.

Q2. Why did you remove outliers?
A2. Outliers can bias the autoencoder and ML models, leading to unstable latent representations and inflated error metrics.

Q3. Why did you scale features using StandardScaler?
A3. Scaling ensures that all features contribute equally during latent representation learning and prevents large-magnitude features from dominating.

Q4. Why a 15% held-out test set?
A4. It balances having enough data for training while reserving an unbiased evaluation set for final performance metrics.

2. Autoencoder & Feature Engineering

Q5. Can you explain the architecture of your attention-based autoencoder?
A5. Input features → Reshape → Dense → Multi-Head Attention → LayerNorm → Dense layers → latent vector → decoder → reconstruction + regression outputs. Attention captures inter-feature relationships.

Q6. Why did you concatenate latent features with original features?
A6. To retain original informative signals while incorporating nonlinear latent patterns, enhancing regression performance.

Q7. How did you choose the latent dimension (64)?
A7. I empirically tested multiple dimensions (32, 64, 128) and 64 offered a balance between compression and information retention.

Q8. Why did you apply PCA after concatenation?
A8. PCA reduces redundancy and noise in the combined features, retaining 95% variance for efficient modeling.

3. Machine Learning Models

Q9. Why did you choose tree-based models like CatBoost, XGB, ExtraTrees?
A9. They handle nonlinearities, high-dimensional data, and feature interactions efficiently, often outperforming linear models.

Q10. Why include linear models like ARDRegression?
A10. To compare performance against simpler baselines and assess linear relationships in the data.

Q11. Why nested 10-fold cross-validation?
A11. It provides unbiased estimation of generalization error, with inner folds for hyperparameter tuning and outer folds for evaluation.

Q12. How did you avoid overfitting in tree-based models?
A12. By limiting max_depth, using subsampling, regularization parameters, and validating via nested CV.

4. Metrics & Evaluation

Q13. Why report R², MAE, MSE, SNR, PSNR?
A13. R² shows variance explained, MAE/MSE quantify prediction errors, SNR/PSNR assess signal reconstruction quality relative to noise.

Q14. How is SNR relevant in regression?
A14. SNR quantifies the strength of signal (true motor UPDRS) versus residual noise, indicating prediction reliability.

Q15. What does PSNR indicate?
A15. PSNR measures reconstruction quality in decibels, useful when comparing predicted vs. true values, especially in latent-enhanced features.

Q16. How do you interpret Mean ± Std across folds?
A16. It represents average performance and variability, indicating model stability across different splits.

5. Interpretability

Q17. Why use SHAP or LIME?
A17. To identify which features (original or latent) most influence model predictions, ensuring transparency.

Q18. Can SHAP handle PCA or latent features?
A18. Yes, but interpretation is easier on original features; latent features can be mapped back to original features for understanding contributions.

Q19. Which features contributed most to motor UPDRS predictions?
A19. Both traditional voice features (e.g., jitter, shimmer) and latent representations capturing complex nonlinear patterns.

6. Workflow & Design Choices

Q20. Why this workflow: Scaling → AE → Concatenate → ML?
A20. Scaling normalizes data → AE captures nonlinear latent patterns → concatenation retains original info → ML leverages full enriched feature set.

Q21. Could you skip PCA or feature selection?
A21. Yes, but PCA reduces noise and SFS ensures only relevant features are used, improving generalization and model interpretability.

Q22. What are the limitations of this workflow?
A22. It is computationally intensive, may overfit small datasets, and latent features can be harder to interpret without mapping.

7. Clinical and Broader Implications

Q23. How could this framework be applied clinically?
A23. To predict motor UPDRS from voice recordings noninvasively, enabling remote monitoring of Parkinson’s disease progression.

Q24. Could this approach generalize to other biomedical datasets?
A24. Yes, any high-dimensional time-series or tabular biomedical data where latent patterns improve prediction.


1. Input features → Reshape

Purpose: Convert the input feature vector into a shape suitable for attention layers.

Many attention layers expect a 3D input: (batch_size, sequence_length, feature_dim).

Example: Your protein features [F1, F2, …, F9] become [batch_size, 9, 1].

2. Dense → Multi-Head Attention

Dense: Projects each feature to a higher-dimensional space so attention can learn more complex interactions.

Multi-Head Attention:

Captures inter-feature relationships (how one feature depends on others).

“Multi-head” means it looks at relationships from multiple perspectives simultaneously.

Why important: RMSD depends on combined effects of multiple physicochemical features. Attention helps the model focus on critical interactions.

3. LayerNorm

Normalizes the output of the attention layer.

Purpose: Stabilizes training and improves convergence by reducing internal covariate shift.

4. Dense layers → latent vector

Dense layers process the attention output and compress it into a latent vector.

Latent vector (embedding): A compact representation of the input features capturing the most important information.

Why important: Reduces dimensionality, removes noise, and produces features that downstream models (ExtraTrees, CatBoost) can use efficiently.

5. Decoder → Reconstruction output

Decoder tries to reconstruct the original input features from the latent vector.

Purpose: Forces the latent vector to retain maximum information about the input.

This is the autoencoder part of the model.

6. Regression output

A separate dense head predicts RMSD directly from the latent vector.

Purpose: The network learns latent features that are informative for both reconstruction and RMSD prediction.

Loss weighting: You can balance reconstruction vs regression to improve RMSD prediction performance.

Summary of Workflow Purpose

Reshape: Prepare features for attention.

Attention: Capture relationships between features.

LayerNorm: Stabilize learning.

Dense → latent: Compress features into informative embeddings.

Decoder: Preserve information (autoencoder).

Regression output: Predict RMSD using learned latent features.




1. What is multicollinearity?

Multicollinearity occurs when two or more features are highly correlated.

This means they carry redundant information, which can confuse some models.

2. Why do we need to remove it?

a) Improves model stability:

In linear models (e.g., ARDRegression, ElasticNet), multicollinearity can make coefficient estimates unstable.

Small changes in the data can lead to large changes in model coefficients.

b) Reduces redundancy:

Highly correlated features don’t add new information, so removing them simplifies the model without losing predictive power.

c) Helps interpretability:

When features are independent, it’s easier to understand which features truly affect the output (e.g., RMSD).

d) Improves some models’ performance:

Tree-based models like ExtraTrees are less sensitive, but linear models and neural networks may benefit from reduced correlation, improving training speed and convergence.

Example statement for interview:

“We remove multicollinearity to reduce redundant information, stabilize coefficient estimates, and improve model interpretability. This ensures that each feature contributes uniquely to RMSD prediction and prevents 


1. What is scaling?

Scaling (or feature scaling) is the process of resizing the range of features so that they have similar magnitudes.

Many machine learning models perform better if features are on the same scale, especially when using distances (e.g., KNN) or gradients (e.g., neural networks).

Common scaling methods:

Standardization (Z-score): (x - mean) / std → mean = 0, std = 1

Min-Max scaling: Rescales values to a fixed range, usually [0,1]

2. Why did you scale features using StandardScaler?

Your dataset contains features like surface area, mass, distances, penalties, which have different ranges.

Without scaling:

Features with larger magnitudes (e.g., molecular mass) could dominate the model, reducing the influence of smaller-scale features.

Neural networks (like your attention-based autoencoder) might converge slower or get stuck in poor minima.

StandardScaler ensures:

All features have mean = 0 and std = 1

Model treats all features equally, improving training stability and prediction accuracy.models from overemphasizing correlated inputs
Q25. How would you improve this workflow for a larger dataset?
A25. Use deeper autoencoders, hyperpar1. Conceptual Questions

“What is the difference between PCA and LDA?”

Expect to explain: unsupervised vs supervised, variance vs class separation, when to use each.

“Why would you use PCA before training a model?”

Answer: to reduce dimensionality, remove noise, avoid overfitting, speed up computation.

“Can you use LDA for regression?”

Answer: No, LDA is for classification because it maximizes class separability.

2. Application Questions

“How would you apply PCA to your RMSD prediction project?”

Example answer: reduce correlated physicochemical features before feeding them into the autoencoder or tree models.

“If your dataset had labels like stable vs unstable proteins, how would LDA help?”

Example answer: LDA could reduce dimensions while emphasizing features that separate stable from unstable proteins.

3. Interpretation Questions

“You did PCA and got 90% variance explained by 10 components. What does this mean?”

Answer: Most of the data information is captured by 10 principal components, reducing dimensionality without losing much info.

“Why not use PCA and LDA together?”

Answer: You can do PCA first to reduce noise, then LDA to maximize class separation if needed, but for regression (like RMSD) LDA is not used.

4. Practical/Implementation Questions

“How do you decide the number of components in PCA?”

Scree plot, cumulative variance explained (e.g., 95% variance).

“Which Python libraries can you use for PCA or LDA?”

sklearn.decomposition.PCA and sklearn.discriminant_analysis.LinearDiscriminantAnalysis.

“Does PCA always improve model performance?”

Not always. PCA reduces dimensionality but may remove features important for prediction.




1. Challenges Faced

a) High-dimensional features

Your protein dataset has many physicochemical descriptors (F1–F9, RMSD, derived features).

Problem: High-dimensional data can lead to overfitting and slower training.

b) Feature correlation / multicollinearity

Many features are correlated (e.g., surface area vs non-polar area).

Problem: Can destabilize linear models and confuse feature importance.

c) Complex non-linear relationships

RMSD depends on combined effects of multiple features, not just individual ones.

Problem: Simple models like linear regression may underperform.

d) Small dataset

Limited number of protein decoys may reduce generalization.

2. How You Overcame Them

a) Dimensionality reduction & feature engineering

Used attention-based autoencoder to extract latent features that capture most important information.

b) Scaling & preprocessing

Standardized features using StandardScaler to ensure all features contribute equally.

Removed outliers and addressed multicollinearity to improve stability.

c) Advanced modeling

Tested multiple tree-based and regression models (ExtraTrees, CatBoost, LGBM, ARDRegression, etc.).

Attention mechanism allowed the model to capture inter-feature relationships that simple models cannot.

d) Evaluation & validation

Used 10-fold CV and a held-out test set to ensure robust performance.

Metrics like R², MAE, SNR, PSNR were tracked to quantify performance and reliability.

3. Novelty of Your Work

Attention-based autoencoder for protein features

Combines representation learning with regression, capturing inter-feature interactions.

Hybrid approach

Latent features from autoencoder + original features → improve regression models like ExtraTrees.

Comprehensive evaluation

Multiple metrics (R², MAE, MSE, SNR, PSNR) + nested 10-fold CV ensures robustness.

Biological relevance

Prediction of RMSD informs about protein structure quality, which is critical for understanding ligand binding and protein stability.



1. Challenges Faced

a. High-dimensional, noisy biomedical voice features

Many features (jitter, shimmer, RPDE, etc.) are correlated and contain noise.

Risk: Overfitting ML models and unstable latent representations.

b. Limited sample size per subject

Only ~200 recordings per patient, 42 patients.

Risk: Small dataset may not generalize well to unseen data.

c. Nonlinear relationships between voice features and motor UPDRS

Linear models alone cannot capture complex dependencies.

d. Model interpretability

Latent features extracted by autoencoder are hard to interpret directly.

e. Avoiding data leakage

Performing feature selection, PCA, and autoencoder training without proper splits could lead to inflated performance.

2. How Challenges Were Overcome

a. Dimensionality reduction and latent encoding

Used attention-based autoencoder to capture nonlinear patterns.

Combined original + latent features to retain informative signals.

b. Regularization & careful training

Dropout layers, batch normalization, and early stopping in AE.

Parameter tuning in ML models (max_depth, min_samples_split, learning rate) to avoid overfitting.

c. Nested cross-validation & held-out test set

Prevented data leakage.

Provided unbiased performance metrics (R², MAE, MSE, SNR, PSNR).

d. Feature selection & PCA

Sequential Feature Selector to remove irrelevant/redundant features.

PCA to retain 95% variance and reduce noise.

e. Interpretability using SHAP/LIME

Mapped latent features’ influence back to original features.

Ensured clinical insights and transparency.

3. Novelty / Contribution

a. Latent-Enhanced Autoencoder Framework

Combines attention-based AE latent features with original biomedical features.

Captures complex nonlinear relationships while retaining interpretable signals.

b. Hybrid ML Pipeline

Integration of AE + ML regressors (tree-based & linear) in a nested CV workflow.

Produces robust predictions for motor UPDRS.

c. Comprehensive Evaluation

Multi-metric evaluation: R², MAE, MSE, SNR, PSNR.

Stability assessed across folds and on a held-out test set.

d. Interpretability

Uses SHAP/LIME on enriched features for clinical insight.

Identifies top contributing original voice features and latent patterns.


1. Motivation & Problem Understanding

Q: Why did you choose motor UPDRS prediction from voice data?
A: Motor UPDRS is a key indicator of Parkinson’s disease progression. Voice recordings are noninvasive and easy to collect remotely, enabling early detection and monitoring without hospital visits. Our goal was to leverage machine learning to predict motor UPDRS from voice features accurately.

Q: Why not predict total UPDRS or use other signals?
A: Total UPDRS includes cognitive and ADL components, which may not correlate strongly with voice. Motor UPDRS directly reflects motor symptoms, which are more likely to influence voice characteristics. This improves signal-to-noise in our model.

2. Data & Preprocessing

Q: How did you handle missing data or outliers?
A: Missing values were imputed using median values. Outliers were detected via Z-score thresholds and removed to reduce noise, ensuring stable model training.

Q: Why StandardScaler over other scalers?
A: StandardScaler centers features and scales them to unit variance, which is important for neural network training and PCA, ensuring features contribute equally to distance-based calculations.

Q: How did you split the dataset?
A: We held out 15% as a test set and used the remaining 85% for nested 10-fold cross-validation. This ensures independent evaluation while maximizing training data.

Q: Did you account for patient-level grouping?
A: Yes, we ensured voice recordings from the same patient were consistently in either training or validation folds to prevent data leakage and over-optimistic performance estimates.

3. Model Architecture & Design

Q: Why attention-based autoencoder?
A: Attention mechanisms help the autoencoder focus on important feature interactions. The latent representation captures nonlinear patterns in high-dimensional voice features, improving downstream regression performance.

Q: Why combine original and latent features?
A: Original features retain interpretable biomedical information, while latent features capture hidden nonlinear relationships. Concatenating both creates a richer feature set for ML models.

Q: Why add PCA and feature selection after AE?
A: PCA reduces dimensionality and noise, while sequential feature selection retains the most predictive features. This combination improves model stability and generalization.

4. Machine Learning Models

Q: Why use ExtraTrees, CatBoost, XGB, etc.?
A: Tree-based models handle high-dimensional, correlated features and are robust to outliers. Using multiple regressors allows us to compare performance and select the best model.

Q: Did you tune hyperparameters?
A: Yes, we used cross-validation with grid search to tune parameters like tree depth, number of estimators, and learning rate, balancing bias and variance.

Q: Did you try neural networks for regression?
A: We focused on tree-based models after AE feature extraction, as they provided better interpretability and stability on this dataset size. Neural networks may be explored in future work with larger datasets.

5. Training & Validation

Q: Why nested 10-fold CV?
A: Nested CV prevents information leakage from hyperparameter tuning to validation. The inner loop selects features and tunes models, while the outer loop evaluates generalization.

Q: How did you prevent data leakage?
A: Each CV fold trained the autoencoder, PCA, and feature selection only on training data. Validation and test sets were strictly unseen during feature engineering.

Q: How did you handle overfitting?
A: Autoencoder used dropout, batch normalization, early stopping, and learning rate reduction. Tree-based models used depth and leaf restrictions. Metrics were monitored across folds to detect overfitting.

6. Metrics & Evaluation

Q: Why R², MAE, MSE, SNR, and PSNR?
A: R² measures explained variance, MAE/MSE capture absolute errors, and SNR/PSNR quantify signal fidelity relative to noise—important for biomedical signal predictions.

Q: How reliable are metrics given the small dataset?
A: Nested CV and a held-out test set help ensure stable, unbiased performance estimates despite the dataset size.

7. Interpretability & Explainability

Q: How did you use SHAP/LIME?
A: SHAP values were computed on the final model to quantify each feature’s contribution. LIME provided local explanations for individual predictions. Both original and latent features were interpretable in the context of motor UPDRS.

Q: Which features contributed most?
A: Both voice-derived jitter/shimmer measures and selected latent features had high SHAP values, indicating strong influence on predictions.

8. Challenges & Limitations

Q: What were the main challenges?
A: High-dimensional voice features, small sample size, and risk of data leakage during AE training.

Q: How did you overcome them?
A: Used attention-based AE to compress features, nested CV to prevent leakage, and PCA + feature selection to reduce dimensionality.

Q: Limitations?
A: Dataset size limits generalizability; model is trained on early-stage Parkinson’s only. Latent features are less interpretable biologically.

9. Novelty & Impact

Q: What is novel here?
A: Latent-enhanced AE captures nonlinear voice patterns and improves ML regression performance. Combining original + latent features and using SHAP/LIME for interpretation is novel in motor UPDRS prediction.

Q: How does it impact remote monitoring?
A: Enables noninvasive, accurate, and interpretable prediction of motor UPDRS from home-collected voice data.

10. Future Work

Q: How to improve the model?
A: Increase dataset size, integrate multimodal signals (e.g., motion sensors), and explore deep learning regression models for end-to-end prediction.

e. Potential Clinical Utility

Enables remote, noninvasive prediction of Parkinson’s motor severity using voice data.

Not always. PCA reduces dimensionality but may remove features important for prediction.ameter tuning, distributed training, and more robust feature selection techniques.
