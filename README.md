# Smart Irrigation Monitoring System

## Project structure
- `data/simulated_irrigation_data.csv`: 5,000 synthetic sensor observations with a useful mix of positive and negative labels
- `train_model.py`: trains and evaluates a Random Forest classifier
- `app.py`: Streamlit monitoring dashboard and interactive prediction
- `models/`: trained model is saved here after training

## Run on Ubuntu
```bash
cd smart_irrigation_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset columns
- `timestamp`: simulated observation time
- `temperature_c`: air temperature in °C
- `humidity_pct`: relative humidity in %
- `light_lux`: light intensity in lux
- `soil_moisture_pct`: soil moisture percentage
- `irrigation_required`: target label (1 = irrigation recommended, 0 = not required)

## Important
This is synthetic educational data. The labels are generated from a rule-inspired probabilistic simulation, not from agronomist-labelled field measurements. Model metrics on this dataset are not evidence of real-world irrigation performance. Calibrate thresholds and validate using local soil, crop, sensor, and field data before any deployment. The pump signal in the app is simulated only; it does not operate hardware.
