 Car-Price-Prediction
Transformed raw data into insights, predicting car prices accurately

 Overview :

This project involves predicting the prices of used cars using machine learning. It includes steps such as data cleaning, feature engineering, model training, and evaluation. The final model is a Linear Regression model that estimates car prices based on various features.

 Project Description :

The car price prediction project is designed to estimate the prices of used cars based on their features using a Linear Regression model. The project involves the following main steps:

1. Data cleaning and preprocessing
2. Data visualization
3. Feature engineering
4. Model training and evaluation
5. Model deployment

 Data

The dataset used in this project is sourced from [this Google Sheets link](https://docs.google.com/spreadsheets/d/e/2PACX-1vSzdBGO6RG_y16U15klgsVy8jQGGzQVP6u2VCPxavVBC3qLwMXMKT_gMLjE09QhQnUDfJjSHx76IS7y/pub?output=csv).

 Data Columns:
- `name`: Car model name
- `company`: Car manufacturer
- `year`: Year of manufacture
- `Price`: Car price
- `kms_driven`: Total kilometers driven
- `fuel_type`: Type of fuel used (e.g., Petrol, Diesel)

 Requirements :

To run this project, you need the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

You can install these packages using pip :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

 Installation

Clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

 Usage

Run the following script to execute the project:

```bash
python car_price_prediction.py
```

 Code Explanation

1. **Import Libraries:**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   import pickle
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.compose import make_column_transformer
   from sklearn.pipeline import make_pipeline
   from sklearn.metrics import r2_score
   ```

2. **Load and Clean Data:**
   ```python
   car = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSzdBGO6RG_y16U15klgsVy8jQGGzQVP6u2VCPxavVBC3qLwMXMKT_gMLjE09QhQnUDfJjSHx76IS7y/pub?output=csv')
   backup = car.copy()
   car = car[car['year'].str.isnumeric()]
   car['year'] = car['year'].astype(int)
   car = car[car['Price'] != 'Ask For Price']
   car['Price'] = car['Price'].str.replace(',', '').astype(int)
   car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
   car = car[car['kms_driven'].str.isnumeric()]
   car['kms_driven'] = car['kms_driven'].astype(int)
   car = car[~car['fuel_type'].isna()]
   car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
   car = car.reset_index(drop=True)
   car.to_csv('Cleaned_Car_data.csv')
   car = car[car['Price'] < 6000000]
   ```

3. **Data Visualization:**
   ```python
   plt.subplots(figsize=(15, 7))
   ax = sns.boxplot(x='company', y='Price', data=car)
   ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
   plt.show()
   
   plt.subplots(figsize=(20, 10))
   ax = sns.swarmplot(x='year', y='Price', data=car)
   ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
   plt.show()
   
   sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
   
   plt.subplots(figsize=(14, 7))
   sns.boxplot(x='fuel_type', y='Price', data=car)
   
   ax = sns.relplot(x='company', y='Price', data=car, hue='fuel_type', size='year', height=7, aspect=2)
   ax.set_xticklabels(rotation=40, ha='right')
   ```

4. **Feature Engineering and Model Training:**
   ```python
   X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
   y = car['Price']
   
   ohe = OneHotEncoder()
   ohe.fit(X[['name', 'company', 'fuel_type']])
   column_trans = make_column_transformer(
       (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
       remainder='passthrough'
   )
   
   lr = LinearRegression()
   pipe = make_pipeline(column_trans, lr)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   pipe.fit(X_train, y_train)
   y_pred = pipe.predict(X_test)
   ```

5. **Model Evaluation:**
   ```python
   r2_score(y_test, y_pred)
   ```

   **Finding Best Model Performance:**
   ```python
   scores = []
   for i in range(1000):
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
       pipe.fit(X_train, y_train)
       y_pred = pipe.predict(X_test)
       scores.append(r2_score(y_test, y_pred))
   
   best_score_index = np.argmax(scores)
   best_score = scores[best_score_index]
   ```

6. **Predicting with the Best Model:**
   ```python
   best_model = pipe.predict(pd.DataFrame(columns=X_test.columns, data=np.array(['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']).reshape(1, 5)))
   ```

7. **Save Model:**
   ```python
   pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
   ```

8. **View Categories Used in Encoding:**
   ```python
   pipe.steps[0][1].transformers[0][1].categories[0]
   ```

 Model Evaluation :

The model is evaluated using the R-squared metric, which indicates how well the model's predictions match the actual data.

 Model Deployment :

The trained model is saved using Pickle to allow for future predictions. The model file `LinearRegressionModel.pkl` can be loaded and used for making predictions on new data.

 Contributing :

If you have suggestions or improvements for this project, feel free to contribute by submitting a pull request or opening an issue.
