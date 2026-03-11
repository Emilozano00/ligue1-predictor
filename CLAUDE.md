# Liga MX Predictor - Contexto del Proyecto

## Qué es este proyecto
Aplicación web de predicciones para Liga MX usando Machine Learning. Desplegada en Streamlit Cloud.

## Stack
- **Repo:** github.com/Emilozano00/sports-analytics (branch: main)
- **API:** API-Football Pro ($19/mes, 7,500 req/día, League 262) - vigente hasta marzo 2026
- **Frontend:** Streamlit Cloud (redespliega automático en push a main)
- **ML:** scikit-learn (RandomForest), joblib para serialización
- **Datos:** 314 partidos (Ap2025 + Cl2025 + Cl2026 J1-7)

## Estructura clave
```
sports-analytics/
├── src/
│   ├── app.py              → UI Streamlit (1,083 líneas)
│   ├── predict.py           → Modelo resultados 1-X-2 (775 líneas)
│   ├── predict_props.py     → Modelo tarjetas + corners + tiros (757 líneas)
│   ├── extract_season.py    → Descarga fixtures API-Football
│   ├── extract_odds.py      → Descarga odds casas de apuestas
│   ├── update_data.py       → Pipeline: descarga → rebuild → git push
│   └── api_client.py        → Cliente HTTP API-Football
├── data/
│   ├── model.joblib          → Modelo resultados serializado
│   ├── cards_model.joblib    → Modelo tarjetas serializado
│   └── raw/                  → CSVs por temporada + odds + fixtures cache
```

## Modelos activos

### Modelo A - Resultado del partido (1-X-2)
- **Archivo:** src/predict.py → train_model() / retrain_and_save()
- **Algoritmo:** RandomForestClassifier (200 árboles, depth=3, min_samples_leaf=5)
- **Features:** 23 base + 3 odds (odds_prob_home, odds_prob_draw, odds_prob_away)
- **Target:** 0=Local, 1=Empate, 2=Visita
- **Métricas walk-forward (train: Ap2025+Cl2025 J1-7, test: Cl2025 J8-13):**
  - Overall: 57% (31/54)
  - Confianza ALTA (>55%): **92% (12/13)** ← número clave
  - Confianza MEDIA (45-55%): 44% (11/25) ← coin flip
  - Confianza BAJA (<45%): 50% (8/16)
  - Local: 81% (25/31) | Empate: 0% (0/7) | Visita: 38% (6/16)
- **CV 5-fold:** 52.5%

### Modelo B - Tarjetas amarillas (Over/Under 3.5)
- **Archivo:** src/predict_props.py → train_cards_model() / retrain_cards()
- **Algoritmo:** RandomForestRegressor (200 árboles, depth=5, min_samples_leaf=5)
- **Features:** 3 (ref_yc_avg 55%, home_yc_avg 23%, away_yc_avg 22%)
- **Métricas:** O/U 3.5 = 64%, MAE = 1.87
- **Baseline:** 62% (siempre over) → solo +2pp de edge
- **NOTA:** Mercado de tarjetas tiene riesgo de integridad/amaño en Liga MX. Feature principal (árbitro) es potencialmente corruptible.

### Modelos descartados
- **Corners (9.5):** 44% accuracy, MAE 2.99 > baseline 2.76
- **Tiros (24.5):** 53% accuracy, MAE 5.67 > baseline 5.45

## Debilidades conocidas
- El modelo NO predice empates (0/7)
- Visitas solo 38% accuracy
- Confianza MEDIA es básicamente coin flip
- Paper trading NO implementado (predicciones no se guardan)

## App Streamlit - 8 secciones
1. Header Hero (branding Liga MX Clausura 2026)
2. Jornada completa (9 partidos, predicciones resumidas)
3. Predicción individual (barra tricolor, badge confianza)
4. Contexto del partido (racha, goles, posición)
5. Análisis del árbitro (clasificación, predicción tarjetas)
6. Tabla de posiciones
7. Historial de aciertos (backtest desglosado)
8. Footer (disclaimer)

## Restricciones técnicas Streamlit Cloud
- No sidebar
- unsafe_allow_html=True NO funciona dentro de st.columns()
- No se puede usar st.divider(), hide_index, use_container_width (versión vieja)

## Convenciones importantes
- Torneo actual: **Clausura 2026** (no 2025) - Liga MX nombra Clausura por el año calendario
- API Season para Clausura 2026 = 2025
- Siempre hacer commit + push después de cambios (Streamlit redespliega automático)
- El "92% cuando dice ALTA" es el número clave para comunicar a usuarios

## Último commit
- Hash: 0ccc9e6
- Contenido: Reentrenamiento de modelo de tarjetas con datos Clausura 2026
