# Store Sales Prediction: Task 1 & Task 2

Welcome to the **Store Sales Prediction** project! This project is divided into two tasks: **Task 1** focuses on time series analysis and feature engineering, while **Task 2** involves building predictive models using machine learning and deep learning techniques.

The goal of this project is to predict store sales for the next 6 weeks to help the company plan ahead and optimize its resources.

## Table of Contents

- [Task 1: Time Series Preprocessing](#task-1-time-series-preprocessing)
  - [Preprocessing Steps](#preprocessing-steps)
  - [Feature Engineering](#feature-engineering)
  - [Time Series Analysis](#time-series-analysis)
- [Task 2: Building Predictive Models](#task-2-building-predictive-models)
  - [Preprocessing](#preprocessing)
  - [Machine Learning Model: Random Forest Regressor](#machine-learning-model-random-forest-regressor)
  - [Loss Function Selection](#loss-function-selection)
  - [Post Prediction Analysis](#post-prediction-analysis)
  - [Deep Learning Model: LSTM](#deep-learning-model-lstm)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

---

## Task 1: Time Series Preprocessing

### Preprocessing Steps

In Task 1, we preprocess the data to convert raw store sales data into a format suitable for machine learning. The main steps include:

1. **Handling Non-Numeric Data**: We convert categorical data such as `Store` and `DayOfWeek` into numeric values using one-hot encoding or label encoding.
2. **Missing Data**: Missing values are handled by using imputation techniques to fill gaps in the data.
3. **Feature Engineering**: Additional features are derived from datetime columns such as:
   - Weekdays and weekends
   - Days to/from holidays
   - Beginning, mid, and end of the month

### Feature Engineering

Features like **days to/from holidays**, **weekend flags**, and **monthly segmentation** are extracted to better capture the patterns in store sales. This additional information helps the machine learning model make better predictions.

### Time Series Analysis

We perform exploratory data analysis (EDA) to understand sales trends, seasonality, and any patterns that can aid in prediction. The Augmented Dickey-Fuller (ADF) test is used to check the stationarity of the data. If the data is non-stationary, we apply differencing.

---

## Task 2: Building Predictive Models

### Preprocessing

For Task 2, we apply all the preprocessing steps outlined in Task 1 to make the data ready for modeling. The data is scaled using a **StandardScaler** to improve model performance, especially when dealing with algorithms that use Euclidean distances.

### Machine Learning Model: Random Forest Regressor

In this task, we train a **RandomForestRegressor** model using a scikit-learn pipeline. The pipeline ensures that the preprocessing and modeling steps are modular and reproducible. The model is trained on features such as the number of customers, days to holidays, and day of the week, and the sales predictions are evaluated using **Mean Squared Error (MSE)**.

- **Feature Importance**: We analyze the importance of different features in predicting sales using Random Forest’s built-in feature importance.

### Loss Function Selection

For regression problems, the **Mean Squared Error (MSE)** is chosen as the loss function because it heavily penalizes larger errors and helps in minimizing the overall error across the predictions. This approach is standard for regression tasks, but other loss functions such as **Mean Absolute Error (MAE)** or **Huber loss** could be considered based on the problem's needs.

### Post Prediction Analysis

After obtaining the predictions, we perform an analysis to evaluate the confidence intervals of the predictions. This analysis helps to understand the uncertainty in the model's predictions and provides a range for the predicted sales.

### Deep Learning Model: LSTM

For more advanced prediction capabilities, we build a **Long Short-Term Memory (LSTM)** model using **TensorFlow**. LSTMs are particularly good for time series prediction because they can capture long-term dependencies in the data. The following steps are performed:

1. **Stationarity Check**: We check if the time series is stationary and difference the data if necessary.
2. **Sliding Window Technique**: The time series is transformed into supervised learning data by creating features for a sliding window approach.
3. **LSTM Model**: We train the LSTM model with two layers, keeping the computational requirements manageable for Google Colab.
4. **Model Evaluation**: The LSTM model is evaluated, and the model is serialized for future predictions.

---

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.x
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `matplotlib`
  - `seaborn`
  - `statsmodels`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/store-sales-prediction.git
   cd store-sales-prediction
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing and modeling scripts:
   ```bash
   python preprocess_data.py
   python train_models.py
   ```

4. You can then explore the models and results in the `models` directory, which includes serialized models for later use.

---

## Contributing

We welcome contributions to improve the accuracy and efficiency of this project. Please feel free to submit issues and pull requests. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to your fork (`git push origin feature-name`).
5. Submit a pull request.

---


### Conclusion

This project provides a comprehensive approach to predicting store sales, incorporating feature engineering, machine learning models, and deep learning techniques. By leveraging time series data and building both traditional and deep learning models, this solution aims to offer accurate sales predictions that can help businesses plan effectively.

Feel free to reach out if you have any questions or feedback. We hope you find this project helpful!

