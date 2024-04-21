# Forecasting Stock Price Movements: A Comparative Analysis of Machine Learning Techniques

This repository provides Python code for forecasting stock price movements using various machine learning techniques. The analysis includes the implementation and comparison of different models to predict stock prices.

## Dependencies

To run the code in this repository, you will need to install the following libraries:

- [yfinance](https://pypi.org/project/yfinance/): A library to fetch historical market data from Yahoo Finance.
- [matplotlib](https://matplotlib.org/): A comprehensive library for creating static, animated, and interactive visualizations in Python.
- [mplfinance](https://pypi.org/project/mplfinance/): A package for plotting financial data in Matplotlib.
- [numpy](https://numpy.org/): Fundamental package for scientific computing with Python.
- [pandas](https://pandas.pydata.org/): A powerful data analysis and manipulation library.
- [seaborn](https://seaborn.pydata.org/): A Python data visualization library based on matplotlib.
- [scikit-learn](https://scikit-learn.org/stable/): A simple and efficient tool for data mining and data analysis.

## Models Implemented

The following machine learning models are implemented in this repository:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)** with various kernels: Polynomial, Radial Basis Function (RBF)
- **Decision Tree**
- **Random Forest**
- **Artificial Neural Network (ANN)**
- **Bayesian Learning**
- **Perceptron**
- **Regression**
- **XGBoost**

## Usage

To use this code, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Python scripts for each model to observe and compare their performance in forecasting stock price movements.

---

**Background:**
A stock market, or share market, is the aggregation of buyers and sellers of stocks (also called shares). These stocks represent ownership claims on businesses. The fluctuations in stock market values have significant implications for stakeholders’ profitability. Investors aim to purchase stocks at lower prices and sell them at higher prices to maximize profits. Predicting future market movements has long captivated individuals with its blend of adventure, allure, and financial risk. Researchers from various disciplines, including business and computer science, have delved into stock market prediction, employing diverse methodologies and algorithms to analyze market dynamics.

**Machine Learning Approach:**
Machine learning algorithms present promising avenues for forecasting market movements by utilizing historical data and discerning underlying patterns. In this study, we investigate the efficacy of various machine learning techniques in predicting stock prices using a comprehensive array of features extracted from historical stock data. Through empirical analysis and evaluation, we aim to examine the effectiveness of these techniques in navigating the complexities of stock market prediction and possibly gain insights that may empower investors and stakeholders in their decision-making processes.

**Data Preparation:**
A dataset was extracted from yfinance for a specific stock over a defined period. This dataset encompassed a multitude of historical data points, including open, high, low, close prices, and trading volumes. From this dataset, we calculated a range of technical indicators such as MACD, MA, RSI, %K, %D, %B, Volume MA, Volume ROC, ATR, volatility indicators, %R, and candlestick patterns. Each indicator was computed using appropriate formulas tailored to capture crucial trends and patterns in the data.

**Model Training and Deployment:**
The dataset was further partitioned into training and testing subsets to facilitate model training and evaluation. Various machine learning models, including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Artificial Neural Network (ANN), and Naive Bayes classifier (Gaussian), were trained on the feature-rich training dataset to learn the underlying patterns and relationships between features and labels. Once trained, these models were deployed to predict the future or real-time trend of the stock market.

By leveraging the learned insights from the training data, the models aimed to forecast the direction of future price movements, thereby assisting investors and stakeholders in making informed decisions in the dynamic stock market environment.