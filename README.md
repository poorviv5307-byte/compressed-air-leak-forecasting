# AI-Based Compressed-Air Leakage Prediction and Consumption Forecasting

A simulation study that detects compressed-air leaks and forecasts short-term air demand. Real plant data was not available, so the project builds a synthetic 60-day plant with known leak events, which lets each method be checked against the true answer.

## What the project does

1. **Simulates a compressed-air plant** at 15-minute resolution, with a two-shift production schedule, sensor noise, and leaks that grow from 1.2 to 2.5 and then 4.0 m³/min.
2. **Detects leaks** using nightly minimum flow (01:00 to 04:00, when no production runs). An alert fires when the nightly estimate exceeds a baseline threshold on two consecutive days. An Isolation Forest is included for comparison.
3. **Forecasts air flow one hour ahead** with a gradient-boosting model using lagged and calendar features. It is compared with persistence and seasonal naive baselines.
4. **Estimates the annual cost** of the extra leakage in rupees.
**[Open the interactive results page](https://poorviv5307-byte.github.io/compressed-air-leak-forecasting/interactive_results.html)**

## Results

| Task | Method | Result |
| --- | --- | --- |
| Leak detection | Night-flow threshold | Alert on day 21 (leak began day 20), F1 = 0.99 |
| Leak detection | Isolation Forest | F1 = 0.43 |
| Leak size | Estimated from last 20 days | 2.79 m³/min (true: 2.80) |
| Forecast (1 h ahead) | Gradient boosting | MAE 0.28 m³/min, MAPE 3.46% |
| Forecast (1 h ahead) | Persistence | MAE 0.98 m³/min |
| Forecast (1 h ahead) | Seasonal naive | MAE 1.92 m³/min |
| Annual leak cost | Assumed ₹8/kWh, 70% operation | About ₹1.23 lakh per year |

## Repository contents

- `compressed_air_sim.py`: the full simulation, detection, and forecasting script
- `results.json`: all metrics from the run
- `daily_leak_detection.csv`: nightly leak estimates for each day
- `simulated_plant_data_15min.csv`: the generated 15-minute dataset
- `plots/`: four figures (flow zoom, leak detection, forecast, pressure and power)

## Requirements

- Python 3.9 or newer
- pandas, numpy, scikit-learn, matplotlib

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to run

```bash
python compressed_air_sim.py
```

The script prints the results, saves `results.json`, the CSV files, and the plots. The random seed is fixed, so every run gives the same numbers.

## Assumptions and limitations

- The data is synthetic. Real leaks grow gradually, sensors drift, and compressor efficiency varies with load.
- The power model is linear and represents a single compressor.
- The cost estimate depends on an assumed tariff and operating fraction, so it is indicative only.
- The forecast horizon is one hour. Longer horizons are future work.

## Author

Poorvika V,Department Robotics And AI,Dayananda Sagar University.

## Team

- Poorvika V (Roll No. ENG25RA0013)
- Janya P (Roll No. ENG25RA0010)
- Raksha Aruna (Roll No. ENG25RA0014)
- Anjana V M (Roll No. ENG25RA0018)
