# Fast Food Calorie Shock: Food vs. Workout Burn Analysis

> A comprehensive data analysis and machine learning project exploring the caloric impact of fast food meals versus the physical effort required to burn them off.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Data Structure](#data-structure)
- [Analysis & Visualizations](#analysis--visualizations)
- [Model Training](#model-training)
  - [Machine Learning](#machine-learning)
  - [Deep Learning](#deep-learning)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

##  Project Overview

This project analyzes **8,000 fast food meal orders** across multiple chains, examining the caloric content, macronutrient breakdown, and the corresponding **exercise time required** to offset the consumed calories. The analysis reveals the shocking reality of fast food consumption and the physical effort needed to maintain caloric balance.

The project combines **Exploratory Data Analysis (EDA)**, **Machine Learning regression**, and **Deep Learning neural networks** to predict net caloric surplus and understand the relationship between dietary choices and fitness outcomes.

---

##  Dataset Description

**Source:** Kaggle — *Fast Food Calorie Shock: Food vs. Workout Burn*

| Attribute | Details |
|-----------|---------|
| **File Name** | `fast_food_cheat_meal_workout_burn.csv` |
| **Records** | 8,000 |
| **Features** | 23 columns |
| **Missing Values** | 0 |
| **Duplicates** | 0 |

---

##  Features

### Meal Information

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | object | Unique order identifier |
| `fast_food_chain` | object | Restaurant chain name |
| `meal_category` | object | Category of the meal (e.g., Late-Night Value Menu Binge) |
| `item_count` | int64 | Number of items in the order |
| `total_calories` | int64 | Total caloric content of the meal |

### Nutritional Breakdown

| Column | Type | Description |
|--------|------|-------------|
| `fat_grams` | int64 | Total fat content (grams) |
| `saturated_fat_grams` | int64 | Saturated fat content (grams) |
| `carb_grams` | int64 | Carbohydrate content (grams) |
| `sugar_grams` | int64 | Sugar content (grams) |
| `protein_grams` | int64 | Protein content (grams) |
| `sodium_mg` | int64 | Sodium content (milligrams) |

### Consumer Demographics

| Column | Type | Description |
|--------|------|-------------|
| `consumer_gender` | object | Gender of the consumer |
| `consumer_age` | int64 | Age of the consumer |
| `consumer_weight_kg` | float64 | Weight of the consumer (kg) |
| `fitness_level` | object | Fitness level classification |

### Workout & Caloric Metrics

| Column | Type | Description |
|--------|------|-------------|
| `thermic_effect_calories` | float64 | Calories burned through thermic effect |
| `net_caloric_surplus` | float64 | Net caloric surplus after thermic effect |
| `walking_burn_time_min` | float64 | Minutes of walking to burn the meal |
| `cycling_burn_time_min` | float64 | Minutes of cycling to burn the meal |
| `running_burn_time_min` | float64 | Minutes of running to burn the meal |
| `hiit_burn_time_min` | float64 | Minutes of HIIT to burn the meal |
| `daily_calorie_allowance_pct` | float64 | Percentage of daily calorie allowance consumed |
| `workout_shock_tier` | object | Classification tier for workout shock level |

---

##  Data Structure

### Sample Record
- order_id: FFC-00001
- fast_food_chain: McDonald's
- meal_category: Late-Night Value Menu Binge
- item_count: 3
- total_calories: 1015
- fat_grams: 47
- saturated_fat_grams: 19
- carb_grams: 117
- sugar_grams: 55
- protein_grams: 32
- sodium_mg: 1632
- consumer_gender: Male
- consumer_age: 61
- consumer_weight_kg: 96.0
- fitness_level: Moderately Active
- thermic_effect_calories: 77.9
- net_caloric_surplus: 937.1
- walking_burn_time_min: 168.3
- cycling_burn_time_min: 80.8
- running_burn_time_min: 60.2
- hiit_burn_time_min: 46.5
- daily_calorie_allowance_pct: 50.7
- workout_shock_tier: High Sweat Penalty


### Statistical Summary

| Statistic | item_count | total_calories | fat_grams | saturated_fat_grams |
|-----------|-----------|----------------|-----------|---------------------|
| **count** | 8000 | 8000 | 8000 | 8000 |
| **mean** | 3.50 | 1602.57 | 80.06 | 31.99 |
| **std** | 1.12 | 672.13 | 34.29 | 14.40 |
| **min** | 2.00 | 750.00 | 32.00 | 10.00 |
| **25%** | 2.00 | 1080.00 | 53.00 | 21.00 |
| **50%** | 4.00 | 1479.00 | 74.00 | 29.00 |
| **75%** | 5.00 | 2008.00 | 100.00 | 40.00 |
| **max** | 5.00 | 3600.00 | 190.00 | 75.00 |

| Statistic | carb_grams | sugar_grams | protein_grams | sodium_mg | consumer_age |
|-----------|-----------|-------------|---------------|-----------|--------------|
| **mean** | 165.96 | 62.25 | 54.04 | 2399.54 | 41.48 |
| **std** | 70.79 | 31.80 | 30.69 | 1072.81 | 13.93 |
| **min** | 66.00 | 14.00 | 15.00 | 826.00 | 18.00 |
| **max** | 380.00 | 175.00 | 140.00 | 5800.00 | 65.00 |

| Statistic | consumer_weight_kg | thermic_effect_calories | net_caloric_surplus |
|-----------|-------------------|------------------------|---------------------|
| **mean** | 77.80 | 121.56 | 1481.02 |
| **std** | 13.93 | 52.72 | 622.81 |
| **min** | 48.00 | 47.10 | 675.50 |
| **max** | 129.40 | 291.10 | 3429.20 |

| Statistic | walking_burn_time_min | cycling_burn_time_min | running_burn_time_min |
|-----------|----------------------|----------------------|----------------------|
| **mean** | 350.15 | 165.04 | 117.92 |
| **std** | 164.47 | 77.54 | 55.49 |
| **min** | 100.50 | 42.00 | 26.30 |
| **max** | 1299.20 | 614.60 | 436.50 |

---

##  Analysis & Visualizations

The notebook includes the following key visualizations:

### 1. Average Calorie Load Across Meal Categories
- Bar plot comparing mean `total_calories` across different `meal_category` values
- Rotated x-axis labels for readability

### 2. Relationship Between Dietary Fat and Total Calories
- Scatter plot of `fat_grams` vs `total_calories`
- Color-coded by `meal_category`
- Shows strong positive correlation

### 3. Average Running Time Required to Offset Intake
- Bar plot of mean `running_burn_time_min` grouped by `fitness_level`
- Demonstrates fitness level impact on recovery time

### 4. Average Net Caloric Surplus Across Fast-Food Chains
- Bar plot of mean `net_caloric_surplus` by `fast_food_chain`
- Identifies which chains contribute most to caloric excess

### 5. Average Walking Time Required by Fitness Level
- Bar plot of mean `walking_burn_time_min` grouped by `fitness_level`
- Shows variation in walking burn efficiency

---

##  Model Training

### Target Variable
- **`net_caloric_surplus`** — The primary regression target representing the net caloric surplus after accounting for the thermic effect of food.

### Feature Selection
Relevant numerical features were selected for model training, including:
- `item_count`
- `total_calories`
- `fat_grams`
- `saturated_fat_grams`
- `carb_grams`
- `sugar_grams`
- `protein_grams`
- `sodium_mg`
- `consumer_age`
- `consumer_weight_kg`
- `thermic_effect_calories`

### Train/Test Split
- Standard train/test split applied
- Features scaled using `StandardScaler` (fit on training data only)

---

### Machine Learning Approach

**Algorithm:** Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`)

**Results:**

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9986 |
| **RMSE** | 22.63 |

---

### Deep Learning Approach

**Architecture:** Sequential Neural Network (TensorFlow/Keras)

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output
])
```

### Compilation:

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

### Callbacks:
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

### Training: 300 epochs maximum (with early stopping)
### Results:
| Metric | Value |
|--------|--------|
| R² Score| 0.9991 |
| MAE | 8.67 |
| RMSE| 18.07|

###  Key Findings
- Average meal contains ~1,603 calories with 80g fat, 166g carbs, and 54g protein
- Average walking time to burn a meal: ~350 minutes (~5.8 hours)
- Average running time: ~118 minutes (~2 hours)
- HIIT is most efficient at ~91 minutes average burn time
- Sodium intake is extremely high — averaging 2,399mg per meal (near daily recommended limit)
- Net caloric surplus averages 1,481 calories after thermic effect
- Deep Learning model outperforms traditional ML with R² of 0.9991 vs 0.9986
- Fitness level significantly impacts required burn time across all exercise types
- No missing values or duplicates found in the dataset — clean data quality

### Technologies Used

| Category | Tools |
|--------|--------|
| Language | Python 3.13 |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (GradientBoostingRegressor, StandardScaler, train_test_split) |
| Deep Learning | TensorFlow / Keras |
| Metrics | R² Score, MAE, RMSE |
| Environment | Kaggle Notebooks |


## Getting Started
### Prerequisites
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

### Running the Notebook
1. Clone the repository or download the notebook
2. Ensure the CSV file is in the correct path:
   
      /kaggle/input/datasets/samartalwar/fast-food-calorie-shock-food-vs-workout-burn/fast_food_cheat_meal_workout_burn.csv
3. Feel free to use the dataset in the    
4. Run all cells in order

### Loading the Data
```python
import pandas as pd
df = pd.read_csv('fast_food_cheat_meal_workout_burn') # when downloading from kaggle directly else use the csv name which it was renamed to.
print(df.shape)  # (8000, 23)
print(df.head())
```
##  Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
- Fork the repository
- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

##  License
This project is open source and available under the MIT License.

##  Acknowledgments
Dataset Source: Kaggle — Fast Food Calorie Shock: Food vs. Workout Burn
Analysis Framework: Kaggle Notebooks with GPU support
Libraries: pandas, scikit-learn, TensorFlow/Keras, matplotlib, seaborn

##  Contact
For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

⚠️ Disclaimer: This project is for educational and analytical purposes only. Nutritional values are simulated/approximate. Always consult a healthcare professional for dietary and fitness advice.
