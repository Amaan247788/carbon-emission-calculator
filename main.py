# short haul flights = less than 1500 km
# for short haul flights the emission factor is 0.15 to 0.2 kg CO2/ km / passenger and 0.11 to 0.16 kg for long
# carbon emitted in kg = distance x emission factor
# need to read in data from the json file and calculate the carbon emmited  

import pandas
import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# hash map = {departure iata: Array[Pair<arrival iata, carbon emitted>]}

# Find a way to loop through json file to get every departure iata
def reader() :
    # filename = open('sample_flights.json')
    filename = open('airline_routes.json')
    data = json.load(filename)

    # Create an empty hashmap (dictionary) to store the results
    departure_map = {}

    # Iterate through each airport in the data
    for departure_iata, airport_data in data.items():
        routes = airport_data.get('routes', [])
        destinations = []
        for route in routes:
            destination_iata = route.get('iata')
            distance_km = route.get('km')
            if distance_km < 1500:
                carbon_emitted = distance_km * 0.175
            else:
                carbon_emitted = distance_km * 0.135
            carbon_emitted = round(carbon_emitted, 1)
            if destination_iata and distance_km is not None:
                destinations.append((destination_iata, carbon_emitted))
        departure_map[departure_iata] = destinations

    # Print the hashmap (departure_map)
    # print(departure_map)
    return departure_map

departure_map = reader()


# making of the website

app = dash.Dash(__name__)

dark_theme = {
    "main-background": "#000000",
    "header-text": "#ff7575",
    "sub-text": "#ffd175",
}

light_theme = {
    "main-background": "#ffe7a6",
    "header-text": "#376e00",
    "sub-text": "#0c5703",
}

app.layout = html.Div([
    html.H1("Select Your Departure Airport IATA"),
    html.Label("Departure IATA: "),
    dcc.Dropdown(
        id = 'departure-dropdown',
        options = [{'label': dep, 'value': dep} for dep in departure_map.keys()],
        placeholder = "Select Departure IATA"
    ),

    html.Label("Arrival IATA: "),
    dcc.Dropdown(
        id = 'arrival-dropdown',
        placeholder = "Select Arrival IATA"
    ),
    ],
    style={"backgroundColor": light_theme["main-background"]},
)

@app.callback(
    Output('arrival-dropdown', 'options'),
    Input('departure-dropdown', 'value')
)

def set_arrival_options(selected_departure):
    if selected_departure is None:
        return []
    arrival_options = [{'label': f"{arrival[0]} (Carbon: {arrival[1]} kg)", 'value': arrival[0]} for arrival in departure_map[selected_departure]]
    return arrival_options

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)