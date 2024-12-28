# Housing Price Prediction using Regularized Linear Regression

This repository contains an implementation of a machine learning model to predict housing prices using **Regularized Linear Regression**. The project explores various regularization techniques to improve model performance and reduce overfitting.

## Objective

The goal of this project is to predict the price of a house based on its features, using regularization methods such as **Lasso (L1)** and **Ridge (L2)** regression.

## Project Structure

```
Housing-price-prediction-using-Regularised-linear-regression/
|
├── data/                 # Dataset used for training and testing
├── notebooks/            # Jupyter notebooks for analysis and modeling
├── models/               # Saved models and scripts
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── main.py               # Main script for running the model
```

## Features

- Implements **Ridge** and **Lasso** regression techniques.
- Feature engineering and preprocessing.
- Model evaluation using metrics like **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-Squared**.
- Hyperparameter tuning for optimal regularization.

## Dataset

The dataset contains housing features such as:

- Lot size
- Number of rooms
- Neighborhood quality
- Year built
- Square footage
- And more...

### Data Source
The dataset is sourced from [Kaggle](https://www.kaggle.com/) or other open datasets relevant to housing prices.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SupratimRayChaudhury/Housing-Price-Prediction-Using-Regularized-Linear-Regression.git
```

2. Navigate to the repository:

```bash
cd Housing-price-prediction-using-Regularised-linear-regression
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

Or explore the notebooks for detailed insights:

```bash
jupyter notebook notebooks/
```

## Dependencies

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Results

The model demonstrates:
- Improved accuracy and reduced overfitting with regularization.
- Insights into the importance of housing features.

## Acknowledgments

This project was inspired by the need to create accurate and robust housing price prediction models. Special thanks to the open-source community for providing datasets and tools.

###### config
1. upload the ipynb file to Google colab

2. upload the csv file to Google colab 

3. run all cells





###### graph of error vs lambda in gradient descent
![g](https://raw.githubusercontent.com/SouravG/Housing-price-prediction-using-Regularised-linear-regression/master/download%20(1).png)


###### graph of error vs lambda in Normal equation
![g](https://raw.githubusercontent.com/SouravG/Housing-price-prediction-using-Regularised-linear-regression/master/download.png)
