# Dashboard de Detección de Incendios Forestales

Este proyecto contiene el dashboard y API para la detección y clasificación de riesgo de incendios utilizando modelos de Machine Learning (LightGBM, Random Forest, AdaBoost, SVR, SVM, Gradient Boosting).

## Requisitos Previos
- Python 3.9 o superior

## Configuración y Ejecución

1. Clonar o descargar el repositorio.
2. Abrir la terminal en la carpeta principal del proyecto (`dashboard_incendios`).
3. Crear un entorno virtual:
   ```bash
   python -m venv venv
   ```
4. Activar el entorno virtual:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
5. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
6. Ejecutar la aplicación:
   ```bash
   python app.py
   ```

El servidor local se ejecutará en http://localhost:5000.

## Endpoints de la API
- `/api/predecir-intensidad` (POST): Predice la intensidad del incendio.
- `/api/predecir-dia-noche` (POST): Predice si es de día o de noche en el momento de la detección.
- `/api/clasificar-nivel` (POST): Clasifica el nivel de riesgo del incendio.
