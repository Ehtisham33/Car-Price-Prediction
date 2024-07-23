

# 🚗 Car Price Prediction Project

This repository contains the code and resources for a machine learning project aimed at predicting car prices based on various features. The project utilizes both Linear Regression and Random Forest Regression models to achieve accurate predictions.

## Project Overview

The objective of this project is to develop a predictive model that can accurately estimate car prices based on features such as age, mileage, brand, model, and other relevant attributes. This project showcases the application of machine learning techniques in a real-world scenario.

## Dataset

The dataset used for this project contains the following attributes:
- **name**: Name of the car
- **year**: Year of manufacture
- **selling_price**: Selling price of the car
- **km_driven**: Total kilometers driven
- **fuel**: Type of fuel used (e.g., Petrol, Diesel, CNG)
- **seller_type**: Type of seller (e.g., Dealer, Individual)
- **transmission**: Type of transmission (e.g., Manual, Automatic)
- **owner**: Number of previous owners

## Models Used

1. **Linear Regression:**
   - Implemented as the baseline model.
   - Evaluated using key metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

2. **Random Forest Regressor:**
   - Used to improve prediction accuracy.
   - Conducted cross-validation to ensure model robustness.
   - Performed hyperparameter tuning to optimize model performance.

## Metrics

The performance of both models was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

## Cross-Validation

Cross-validation was conducted to ensure the robustness and reliability of the models. This technique helps in assessing the generalizability of the models to unseen data.

## Hyperparameter Tuning

For the Random Forest model, hyperparameter tuning was performed to find the optimal set of parameters. The following hyperparameters were tuned:
- **n_estimators**: Number of trees in the forest.
- **max_depth**: Maximum depth of the tree.
- **min_samples_split**: Minimum number of samples required to split a node.
- **min_samples_leaf**: Minimum number of samples required at each leaf node.
- **max_features**: Number of features to consider for the best split.

## Repository Structure

- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `scripts/`: Python scripts for data preprocessing, model training, and evaluation.
- `results/`: Contains the evaluation metrics and plots.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Ehtisham33/car-price-prediction.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks in the `notebooks/` directory to explore the data and train the models.

## Conclusion

This project demonstrates the application of machine learning techniques for predicting car prices. The Random Forest model, after hyperparameter tuning, showed significant improvements over the baseline Linear Regression model. 

Feel free to explore the repository and reach out if you have any questions or feedback!

## Connect with Me

- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/ehtisham-yaqoob-161400275/)
- GitHub: [GitHub Profile](https://github.com/Ehtisham33)

---
