import numpy as np
import matplotlib.pyplot as plt
import csv

# Crear arrays vacíos
araba_fiyat = []
araba_max_hiz = []
errors = []

# Abrir el archivo CSV y leer los datos
with open('polynomial-regression.csv', 'r') as archivo_csv:
    lector = csv.DictReader(archivo_csv)
    for fila in lector:
        araba_fiyat.append(float(fila['araba_fiyat']))
        araba_max_hiz.append(float(fila['araba_max_hiz']))

# Convertir las listas en NumPy
araba_fiyat = np.array(araba_fiyat)
araba_max_hiz = np.array(araba_max_hiz)

# Grados de polinomio a probar
degrees = 2

# Implementación de leave-one-out cross-validation
for i in range(len(araba_fiyat)):
    # Separar un punto para validación
    araba_fiyat_val = araba_fiyat[i]
    araba_max_hiz_val = araba_max_hiz[i]

    # Resto de los puntos para entrenamiento
    araba_fiyat_train = np.delete(araba_fiyat, i)
    araba_max_hiz_train = np.delete(araba_max_hiz, i)

    # Ajuste del modelo polinomial
    coefficients = np.polyfit(araba_fiyat_train, araba_max_hiz_train,degrees)
    polynomial = np.poly1d(coefficients)

    # Predecir el punto de validación
    araba_max_hiz_pred = polynomial(araba_fiyat_val)

    # Calcular el error absoluto del modelo
    error = np.abs(araba_max_hiz_val - araba_max_hiz_pred)
    errors.append(error)

# Calcular el error absoluto medio (MAE) para el grado del polinomio
mae = np.mean(errors)

# Visualización de la curva polinómica ajustada
araba_fiyat_plot = np.linspace(min(araba_fiyat), max(araba_fiyat), 100)
araba_max_hiz_plot = polynomial(araba_fiyat_plot)

#Crear el gráfico
plt.title('Regresión Polinomial Simple\nError cuadratico medio: ' + str(mae))
plt.scatter(araba_fiyat, araba_max_hiz)
plt.plot(araba_fiyat_plot, araba_max_hiz_plot, color='red')
plt.legend()
plt.show()