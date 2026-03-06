#from utils import db_connect
#engine = db_connect()

# your code here

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

# 1. CARGA DE DATOS
url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
df = pd.read_csv(url)

# 2. PREPROCESAMIENTO
df_processed = pd.get_dummies(df, drop_first=True)
X = df_processed.drop('charges', axis=1)
y = df_processed['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. OPTIMIZACIÓN (Características Polinómicas)
# Es vital guardar el transformador polinómico además del modelo
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# 4. ENTRENAMIENTO DEL MODELO
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 5. GUARDAR EL MODELO Y EL TRANSFORMADOR
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('poly_features.pkl', 'wb') as file:
    pickle.dump(poly, file)

print("¡Modelo de seguros optimizado y guardado con éxito junto a su transformador!")
