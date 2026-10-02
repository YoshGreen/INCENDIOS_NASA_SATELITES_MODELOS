import os
import json
import traceback
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configurar rutas de modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, 'dashboard_modelos', 'modelos')

# Variables globales para modelos
modelos = {
    'regresion': {},
    'daynight': {},
    'nivel': {}
}
scalers = {}
metricas = {}

def _cargar_modelo(ruta, nombre):
    """Carga un modelo individual con manejo de errores."""
    try:
        modelo = joblib.load(ruta)
        return modelo
    except Exception as e:
        print(f"  [WARN] No se pudo cargar {nombre}: {e}")
        return None

def cargar_modelos():
    global modelos, scalers, metricas
    
    # Scalers
    scalers['regresion'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'scaler_regresion.pkl'), 'scaler_regresion')
    scalers['daynight'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'scaler_daynight.pkl'), 'scaler_daynight')
    scalers['nivel'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'scaler_nivel.pkl'), 'scaler_nivel')
    
    # Modelos de regresion
    modelos['regresion']['lightgbm'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'reg_lightgbm.pkl'), 'reg_lightgbm')
    modelos['regresion']['random_forest'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'reg_random_forest.pkl'), 'reg_random_forest')
    modelos['regresion']['gradient_boosting'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'reg_gradient_boosting.pkl'), 'reg_gradient_boosting')
    modelos['regresion']['adaboost'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'reg_adaboost.pkl'), 'reg_adaboost')
    modelos['regresion']['svr'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'reg_svr.pkl'), 'reg_svr')
    
    # Modelos Dia/Noche
    modelos['daynight']['lightgbm'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_daynight_lightgbm.pkl'), 'clf_daynight_lightgbm')
    modelos['daynight']['random_forest'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_daynight_random_forest.pkl'), 'clf_daynight_random_forest')
    modelos['daynight']['gradient_boosting'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_daynight_gradient_boosting.pkl'), 'clf_daynight_gradient_boosting')
    modelos['daynight']['adaboost'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_daynight_adaboost.pkl'), 'clf_daynight_adaboost')
    modelos['daynight']['svm'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_daynight_svm.pkl'), 'clf_daynight_svm')
    
    # Modelos Nivel
    modelos['nivel']['lightgbm'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_nivel_lightgbm.pkl'), 'clf_nivel_lightgbm')
    modelos['nivel']['random_forest'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_nivel_random_forest.pkl'), 'clf_nivel_random_forest')
    modelos['nivel']['gradient_boosting'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_nivel_gradient_boosting.pkl'), 'clf_nivel_gradient_boosting')
    modelos['nivel']['adaboost'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_nivel_adaboost.pkl'), 'clf_nivel_adaboost')
    modelos['nivel']['svm'] = _cargar_modelo(os.path.join(MODELOS_DIR, 'clf_nivel_svm.pkl'), 'clf_nivel_svm')
    
    # Metricas (cada JSON es una LISTA de dicts, ya ordenada por rendimiento)
    try:
        with open(os.path.join(MODELOS_DIR, 'metricas_regresion.json'), 'r', encoding='utf-8') as f:
            metricas['regresion'] = json.load(f)
        with open(os.path.join(MODELOS_DIR, 'metricas_daynight.json'), 'r', encoding='utf-8') as f:
            metricas['daynight'] = json.load(f)
        with open(os.path.join(MODELOS_DIR, 'metricas_nivel.json'), 'r', encoding='utf-8') as f:
            metricas['nivel'] = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Error al cargar metricas: {e}")
    
    # Verificar que al menos LightGBM cargo correctamente
    lgb_ok = all([
        modelos['regresion'].get('lightgbm'),
        modelos['daynight'].get('lightgbm'),
        modelos['nivel'].get('lightgbm')
    ])
    
    if lgb_ok:
        print("[OK] Modelos LightGBM cargados correctamente. Dashboard listo.")
    else:
        print("[ERROR] FALLO al cargar modelos LightGBM. El dashboard no funcionara.")

cargar_modelos()

@app.route('/')
def inicio():
    return render_template('inicio.html')

@app.route('/predecir-intensidad')
def predecir_intensidad():
    return render_template('predecir_intensidad.html')

@app.route('/api/predecir-intensidad', methods=['POST'])
def api_predecir_intensidad():
    try:
        data = request.json
        features = [
            float(data['brightness']),
            float(data['bright_t31']),
            float(data['scan']),
            float(data['track']),
            float(data['confidence_num'])
        ]
        
        X = np.array(features).reshape(1, -1)
        
        # LightGBM es tree-based, predice directamente sin escalar
        pred_log = modelos['regresion']['lightgbm'].predict(X)[0]
        frp_mw = float(np.expm1(pred_log))
        
        # Asegurar que no sea negativo
        frp_mw = max(frp_mw, 0.0)
        
        if frp_mw < 5:
            nivel = 'bajo'
            nivel_label = 'Bajo Riesgo'
            color = 'success'
            descripcion = 'Intensidad baja. Fuego incipiente o de poca energía.'
        elif 5 <= frp_mw <= 15:
            nivel = 'medio'
            nivel_label = 'Riesgo Medio'
            color = 'warning'
            descripcion = 'Intensidad media. Fuego activo moderado.'
        elif 15 < frp_mw <= 40:
            nivel = 'alto'
            nivel_label = 'Alto Riesgo'
            color = 'danger'
            descripcion = 'Intensidad alta. Fuego activo severo.'
        else:
            nivel = 'critico'
            nivel_label = 'Riesgo Crítico'
            color = 'dark'
            descripcion = 'Intensidad crítica. Gran conflagración extrema.'
            
        return jsonify({
            'frp_mw': round(frp_mw, 2),
            'nivel': nivel,
            'nivel_label': nivel_label,
            'color': color,
            'descripcion': descripcion
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predecir-dia-noche')
def predecir_dia_noche():
    return render_template('predecir_dia_noche.html')

@app.route('/api/predecir-dia-noche', methods=['POST'])
def api_predecir_dia_noche():
    try:
        data = request.json
        features = [
            float(data['brightness']),
            float(data['bright_t31']),
            float(data['frp_log']),
            float(data['scan']),
            float(data['track']),
            float(data['confidence_num'])
        ]
        
        X = np.array(features).reshape(1, -1)
        
        # LightGBM es tree-based, predice directamente sin escalar
        modelo = modelos['daynight']['lightgbm']
        prediccion_val = modelo.predict(X)[0]
        
        probabilidad_dia = None
        probabilidad_noche = None
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(X)[0]
            # Asumimos clase 0=Noche (N), clase 1=Día (D) o según el encoding
            probabilidad_noche = float(probabilidades[0])
            probabilidad_dia = float(probabilidades[1]) if len(probabilidades) > 1 else 1.0 - probabilidad_noche
            
        # Determinar clase predicha ('D' o 'N')
        if prediccion_val == 1 or prediccion_val == 'D' or str(prediccion_val) == '1':
            pred_clase = 'D'
            etiqueta = 'Día'
            color = 'warning'
        else:
            pred_clase = 'N'
            etiqueta = 'Noche'
            color = 'info'
            
        return jsonify({
            'prediccion': pred_clase,
            'etiqueta': etiqueta,
            'probabilidad_dia': round(probabilidad_dia, 4) if probabilidad_dia is not None else None,
            'probabilidad_noche': round(probabilidad_noche, 4) if probabilidad_noche is not None else None,
            'color': color
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/clasificar-nivel')
def clasificar_nivel():
    return render_template('clasificar_nivel.html')

@app.route('/api/clasificar-nivel', methods=['POST'])
def api_clasificar_nivel():
    try:
        data = request.json
        
        # Numeric
        brightness = float(data['brightness'])
        bright_t31 = float(data['bright_t31'])
        scan = float(data['scan'])
        track = float(data['track'])
        confidence_num = float(data['confidence_num'])
        mes = float(data['mes'])
        hora = float(data['hora'])
        
        # Categorical
        daynight = data['daynight']
        satellite = data['satellite']
        instrument = data['instrument']
        
        # One-hot encoding
        daynight_D = 1.0 if daynight == 'D' else 0.0
        daynight_N = 1.0 if daynight == 'N' else 0.0
        
        satellite_Aqua = 1.0 if satellite == 'Aqua' else 0.0
        satellite_N20 = 1.0 if satellite == 'N20' else 0.0
        satellite_N21 = 1.0 if satellite == 'N21' else 0.0
        satellite_SNPP = 1.0 if satellite == 'SNPP' else 0.0
        satellite_Terra = 1.0 if satellite == 'Terra' else 0.0
        
        instrument_MODIS = 1.0 if instrument == 'MODIS' else 0.0
        instrument_SNPP = 1.0 if instrument == 'SNPP' else 0.0
        instrument_VIIRS = 1.0 if instrument == 'VIIRS' else 0.0
        
        # Feature order (17 cols, exacto al training):
        features_num = [brightness, bright_t31, scan, track, confidence_num, mes, hora]
        features_cat = [daynight_D, daynight_N, satellite_Aqua, satellite_N20, satellite_N21,
                        satellite_SNPP, satellite_Terra, instrument_MODIS, instrument_SNPP, instrument_VIIRS]
        
        # Escalar numéricos
        X_num = scalers['nivel'].transform([features_num])[0]
        X_scaled = np.concatenate([X_num, features_cat]).reshape(1, -1)
        
        # El modelo fue entrenado con datos escalados
        modelo = modelos['nivel']['lightgbm']
        prediccion_val = int(modelo.predict(X_scaled)[0])
        
        # Probabilidades como array [bajo, medio, alto]
        probabilidades = []
        if hasattr(modelo, 'predict_proba'):
            probs = modelo.predict_proba(X_scaled)[0]
            probabilidades = [round(float(p), 4) for p in probs]
            
        if prediccion_val == 0:
            etiqueta = 'Bajo'
            color = 'success'
            descripcion = 'Riesgo bajo. Foco de calor de baja intensidad.'
        elif prediccion_val == 1:
            etiqueta = 'Medio'
            color = 'warning'
            descripcion = 'Riesgo medio. Foco de calor de intensidad moderada.'
        else:
            etiqueta = 'Alto'
            color = 'danger'
            descripcion = 'Riesgo alto. Foco de calor de alta intensidad.'
            
        return jsonify({
            'nivel': prediccion_val,
            'etiqueta': etiqueta,
            'color': color,
            'descripcion': descripcion,
            'probabilidades': probabilidades
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/leaderboard')
def leaderboard():
    # Las métricas ya son listas de dicts ordenadas — pasarlas directamente
    return render_template('leaderboard.html',
        metricas_reg=metricas.get('regresion', []),
        metricas_dn=metricas.get('daynight', []),
        metricas_nivel=metricas.get('nivel', [])
    )

@app.route('/anexo-tecnico')
def anexo_tecnico():
    return render_template('anexo_tecnico.html',
        metricas_reg=metricas.get('regresion', []),
        metricas_dn=metricas.get('daynight', []),
        metricas_nivel=metricas.get('nivel', [])
    )

@app.route('/metodologia')
def metodologia():
    return render_template('metodologia.html')

@app.route('/analisis')
def analisis():
    # Extraer feature importance (fijo para no tener que cargarlo lento en cada refresh)
    importancias_features = [
        "Temperatura (brightness)", "Temp Fondo (bright_t31)", "Res. Scan", 
        "Mes", "Res. Track", "Hora", "Confianza %", "Satélite N21", "Otros satélites/instrumentos"
    ]
    importancias_valores = [
        5739, 5360, 4963, 2766, 2450, 2143, 1925, 975, 679
    ]

    return render_template('analisis.html',
                           active_page='analisis',
                           metricas_reg=metricas.get('regresion', []),
                           metricas_dn=metricas.get('daynight', []),
                           metricas_nivel=metricas.get('nivel', []),
                           importancias_features=importancias_features,
                           importancias_valores=importancias_valores)

@app.route('/validacion')
def validacion():
    import json
    
    cv_data = {}
    underfit_data = {}
    complejidad_data = {}
    
    base_dir = os.path.join('COLABS_MODELOS', 'validacion_cruzada')
    
    try:
        with open(os.path.join(base_dir, 'validacion_cruzada.json'), 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
        with open(os.path.join(base_dir, 'underfitting.json'), 'r', encoding='utf-8') as f:
            underfit_data = json.load(f)
        with open(os.path.join(base_dir, 'curva_complejidad.json'), 'r', encoding='utf-8') as f:
            complejidad_data = json.load(f)
    except Exception as e:
        print(f"Error cargando JSON de validacion: {e}")
        
    return render_template('validacion.html', 
                           active_page='validacion',
                           cv_data=cv_data,
                           underfit_data=underfit_data,
                           complejidad_data=complejidad_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
