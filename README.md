# Carbon Emission Calculator

An intelligent carbon emission calculator for flights that uses machine learning to provide accurate carbon footprint estimates and recommendations. This course was started in May 2024, where I developed the backend of the project, parsing over 700,000 lines of json data to calculate carbon emissions from one airport to another. Now, a year later I am trying to integrate ML into the project and develope a better front end interface for the project as a hobby to better my ML and front-end skills. I have listed key criteria and visions of the projects I have below. This is still a work in progress so expect a lot of changes to occur on the daily

## Features

- Flight carbon emission calculation - DONE
- Machine learning-based emission predictions - In progress
- Interactive route visualization - In progress
- Carbon offset recommendations - In progress
- User-friendly interface - Currently a local server that can be run using Dash - but in progress of turning this into a website

## Project Structure

```
carbon-emission-calculator/
src/
    models/          # ML models and training code
    data/            # Data processing and storage
    utils/           # Utility functions
    api/             # API endpoints
tests/               # Unit and integration tests
docs/                # Documentation
requirements.txt     # Project dependencies
README.md            # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/carbon-emission-calculator.git
cd carbon-emission-calculator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python src/main.py
```

## AI/ML Features - Working on these new features in the summer of 2025

- Carbon emission prediction using historical flight data
- Route optimization for minimal carbon impact
- Personalized carbon offset recommendations
- Weather impact analysis on emissions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Amaan Choudhury - amaanc@umich.edu
Project Link: https://github.com/Amaan247788/carbon-emission-calculator 
