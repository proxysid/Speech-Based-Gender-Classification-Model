# Speed-Based Gender Identification Model

This project implements a Machine Learning based Gender Identification Model that predicts the gender and age group of a speaker using voice frequency features extracted from an audio clip.
The system combines signal processing and classification algorithms to achieve accurate predictions based on speech speed and spectral characteristics.

## Project Overview

The Speed-Based Gender Identification Model analyzes recorded voice samples and classifies the speaker’s gender (Male/Female) and age group (Young/Matured/Old).
It uses Random Forest Classifiers trained on preprocessed datasets containing various speech-based features.
The model performs both feature extraction and classification, making it suitable for applications such as human–computer interaction, voice authentication, and demographic analysis.

## Key Features

* Merges cleaned gender and age datasets into a unified dataset.
* Computes correlation heatmaps for feature visualization.
* Trains two separate Random Forest models — one for gender and one for age prediction.
* Extracts voice features such as mean frequency, spectral flatness, roll-off, centroid, etc., using Librosa.
* Records real-time voice input via SoundDevice.
* Predicts both gender and age group from recorded audio.
* Saves and loads trained models using Joblib for future predictions.

## Tech Stack

* **Programming Language:** Python
* **Libraries Used:**

  * pandas — Data manipulation
  * matplotlib, seaborn — Visualization
  * scikit-learn — Machine learning (Random Forest)
  * librosa — Audio feature extraction
  * numpy — Numerical operations
  * sounddevice, scipy.io.wavfile — Audio recording and saving
  * joblib — Model serialization

## Dataset

* **Gender Dataset:** cleaned_gender.csv
* **Age Dataset:** cleaned_age.csv
* Both datasets are merged into a single file named merged_voice_dataset.csv.
* Each record includes acoustic features with corresponding gender and age labels.

Label mappings:

* Gender:

  * 0 → Male
  * 1 → Female
* Age Group:

  * 2 → Young
  * 3 → Matured
  * 4 → Old

## Model Training

1. The merged dataset is split into training (80%) and testing (20%) sets.
2. Two models are trained:

   * gender_model.pkl → Predicts gender
   * age_model.pkl → Predicts age group
3. The models use Random Forest Classifiers with parameters:

   * n_estimators = 100
   * criterion = 'entropy' (for gender)
   * criterion = 'gini' (for age)
4. Model evaluation is done using:

   * Accuracy Score
   * Confusion Matrix
   * Classification Report

## Voice Feature Extraction

The system extracts over 20 acoustic features including:

* Spectral Centroid
* Spectral Flatness
* Spectral Rolloff
* Bandwidth
* Mean, Median, IQR, Skewness, Kurtosis
* Amplitude variation metrics (meanfun, minfun, maxfun)
* Modulation index and entropy

These features form the input for gender and age prediction models.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/speed-based-gender-identification.git
cd speed-based-gender-identification
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn librosa sounddevice scipy joblib
```

### 3. Prepare the dataset

Place `cleaned_gender.csv` and `cleaned_age.csv` in the project directory.

### 4. Train and save models

```bash
python GENDERid.py
```

The script will:

* Merge datasets
* Train models
* Generate evaluation reports
* Save `gender_model.pkl` and `age_model.pkl`

### 5. Record and test prediction

The script records a 5-second voice sample, extracts its features, and predicts:

* Gender
* Gender probabilities
* Age group

## Sample Output

```
Recording...
Recording complete.
Gender Accuracy: 0.92
Predicted Gender: Female
Gender Probabilities: {'Male': 15.37, 'Female': 84.63}
Predicted Age Group: Matured
```

## Visualization

A correlation heatmap is generated to visualize relationships between all extracted features and labels.

```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

## Future Scope

* Integrate deep learning models (e.g., CNNs or LSTMs) for improved accuracy.
* Expand dataset to multiple languages and accents.
* Create a web-based UI for live testing.
* Add pitch, timbre, and emotion analysis.

## Author

**Siddharth Chandra Prabhakar**
Final year B.Tech (Electronics & Communication Engineering) <br>
National Institute of Technology Sikkim

## License

This project is open-source and available under the MIT License.
