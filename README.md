# Smart Irrigation Monitoring System

## Project structure
- `data/simulated_irrigation_data.csv`: 5,000 sensor observations with a useful mix of positive and negative labels
- `train_model.py`: trains and evaluates a Random Forest classifier
- `app.py`: Streamlit monitoring dashboard and interactive prediction
- `models/`: trained model is saved here after training

## Dataset columns
- `timestamp`: simulated observation time
- `temperature_c`: air temperature in °C
- `humidity_pct`: relative humidity in %
- `light_lux`: light intensity in lux
- `soil_moisture_pct`: soil moisture percentage
- `irrigation_required`: target label (1 = irrigation recommended, 0 = not required)


