# Ligue 1 Predictor

Predictor de resultados de la Ligue 1 (Francia) usando Machine Learning para apuestas deportivas.

## Stack

- **API:** [API-Football](https://www.api-football.com/) (League 61)
- **ML:** scikit-learn, XGBoost, LightGBM (stacking ensemble)
- **Frontend:** Streamlit
- **Datos:** 1,600 partidos (2021-2025), 46 features

## Modelo

Stacking ensemble con 4 modelos base (LR + RF + XGBoost + MLP) y meta-learner calibrado.

| Metrica | Valor |
|---------|-------|
| Accuracy | 57.3% |
| F1 macro | 0.422 |
| Log Loss | 0.978 |
| Brier | 0.583 |

## Correr localmente

```bash
# Clonar
git clone <repo-url>
cd ligue1-predictor

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar .streamlit/secrets.toml con tu API key

# Correr app
streamlit run app/streamlit_app.py
```

## Estructura

```
app/streamlit_app.py      -> UI (Predictor, Comparacion, Bankroll)
src/data/fetcher.py       -> Descarga datos API-Football
src/data/preprocessor.py  -> JSON -> parquet
src/features/engineer.py  -> Feature engineering (ELO, rolling, referee, H2H)
src/models/trainer.py     -> Entrena 6 modelos individuales
src/models/stacking.py    -> Stacking ensemble
```

## Streamlit Cloud

1. Conectar repo en [share.streamlit.io](https://share.streamlit.io)
2. Main file path: `app/streamlit_app.py`
3. Python version: 3.11
4. En **Secrets**, agregar:
   ```toml
   API_FOOTBALL_KEY = "tu_api_key"
   ```
