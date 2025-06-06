# Carbon Emission Calculator

A machine learning-based tool for predicting carbon emissions from flight routes. The project uses a Random Forest model to predict carbon emissions based on various flight characteristics.

## Features

- Data preprocessing and feature engineering for flight route data
- Machine learning model for carbon emission prediction
- Model evaluation and visualization tools
- Web interface for easy interaction (coming soon)

## Project Structure

```
Carbon Emission Calculator/
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── base_model.py
│   │   └── emission_model.py
│   ├── utils/
│   │   └── visualization.py
│   └── train.py
├── data/
│   └── airline_routes.json
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/carbon-emission-calculator.git
cd carbon-emission-calculator
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model and generate visualizations:

```bash
python src/train.py
```

This will:
- Load and preprocess the flight route data
- Perform feature engineering
- Train the Random Forest model
- Generate performance visualizations
- Save results in the `training_results` directory

### Model Performance

The current model achieves:
- MSE: ~1.12e-07
- RMSE: ~0.0003
- R2 Score: ~0.9999

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 