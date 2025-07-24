import streamlit as st
import os
import json
import webbrowser
from pathlib import Path
import re
import subprocess
import platform
from PIL import Image
import io
import requests
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------

def get_place_details(place_name, location_context="Norway"):
    """
    Function to get Google Maps link and rating for a place
    Returns a tuple of (maps_url, rating)
    
    Args:
        place_name (str): Name of the place
        location_context (str): Additional location context (e.g., Norway, Bergen)
    
    Returns:
        tuple: (maps_url, rating) - Google Maps URL and rating (out of 5)
    """
    try:
        # Prepare search query
        search_query = f"{place_name} {location_context}"
        search_query = search_query.replace(" ", "+")
        
        # Create Google Maps search URL
        maps_url = f"https://www.google.com/maps/search/{search_query}"
        
        # For demonstration purposes, we'll generate simulated ratings
        # In a real application, you would use the Google Places API or web scraping
        # This is simpler and avoids API key requirements for this demo
        
        # Generate a predictable but seemingly random rating between 3.5 and 4.9
        # Using the length of the place name to create variation
        seed = sum(ord(c) for c in place_name)
        rating = 3.5 + (seed % 14) / 10.0
        if rating > 4.9:
            rating = 4.9
        
        # Return the maps URL and the simulated rating
        return maps_url, round(rating, 1)
    
    except Exception as e:
        print(f"Error getting details for {place_name}: {e}")
        return f"https://www.google.com/maps/search/{place_name.replace(' ', '+')}", None

def get_place_details_batch(places, location_context="Norway"):
    """
    Get details for multiple places at once
    
    Args:
        places (list): List of place names
        location_context (str): Location context to add to searches
        
    Returns:
        dict: Dictionary mapping place names to (url, rating) tuples
    """
    results = {}
    for place in places:
        results[place] = get_place_details(place, location_context)
    return results

def get_location_coordinates(location_name):
    """
    Get coordinates for a location using predefined coordinates
    """
    try:
        # Predefined coordinates for locations
        location_coords = {
            "Nusfjord?": (68.0352, 13.3491),
            "Kjerag Hike": (59.0350, 6.5875),
            "Pulpit Rock": (58.9861, 6.1899),
            "Henningsvær + Fløya": (68.1484, 14.2015),
            "Drive to Geirangerfjorden": (62.1049, 7.0752),
            "Geirangerfjord Cruise": (62.1049, 7.0752),
            "Geirangerfjord → Ålesund": (62.1049, 7.0752),
            "Return Home": (58.8819, 5.6264),  # Stavanger Airport coordinates
            "Fly EVE → Bergen": (60.3943, 5.3259),  # Bergen coordinates
            "Bergen → Ålesund": (62.4722, 6.1524),  # Ålesund coordinates
            "Fly Ålesund → Stavanger": (58.9700, 5.7331),  # Stavanger coordinates
            "Haukland Beach – Uttakleiv Beach – Offersøykammen Hike": (68.1925, 13.5157),  # Haukland Beach
            "Steffenakken – Reinebringen – Hamnøy – Ramberg": (67.9395, 13.1369),  # Hamnøy
            "IAD → FRA → EVE": (68.4891, 16.6780),  # Evenes Airport
            "Evenes → Vågan, Lofoten": (68.2218, 13.3457),  # Lofoten
            "Vågan, Lofoten": (68.2218, 13.3457),  # Lofoten
            "Lofoten": (68.2218, 13.3457),  # Lofoten
            "Bergen": (60.3943, 5.3259),  # Bergen
            "Ålesund": (62.4722, 6.1524),  # Ålesund
            "Stavanger": (58.9700, 5.7331),  # Stavanger
        }
        
        # Check if we have predefined coordinates for this location
        if location_name in location_coords:
            return location_coords[location_name]
            
        # For special cases with airport codes
        if "IAD" in location_name:
            return 38.9531, -77.4565  # Washington Dulles coordinates
        if "FRA" in location_name:
            return 50.0379, 8.5622    # Frankfurt Airport coordinates
        if "EVE" in location_name:
            return 68.4891, 16.6780   # Evenes Airport coordinates
        
        # If no match found, return None
        return None
    except Exception as e:
        print(f"Error getting coordinates for {location_name}: {e}")
        return None

def get_current_weather(lat, lon):
    """
    Get current weather for coordinates using Met Norway API
    This is a simplified version that extracts current weather from the forecast
    """
    try:
        # Get forecast from Met Norway API
        forecast_data = get_forecast(lat, lon)
        
        if forecast_data and 'hourly' in forecast_data and forecast_data['hourly']:
            # First hour is current weather
            current = forecast_data['hourly'][0]
            
            # Format it like OpenWeatherMap data for compatibility with existing UI
            weather_data = {
                'weather': [current['weather'][0]],
                'main': {
                    'temp': current['temp'],
                    'feels_like': current['temp'] - 2,  # Approximate feels like
                    'humidity': current['humidity'],
                    'pressure': 1013  # Default pressure
                },
                'wind': {
                    'speed': current['wind_speed'],
                    'deg': current['wind_deg']
                }
            }
            
            return weather_data
        
        return None
    except Exception as e:
        print(f"Error getting current weather: {e}")
        return None

def get_forecast(lat, lon):
    """
    Get forecast for coordinates using Met Norway API (no API key required)
    """
    try:
        # Format the Met Norway API URL with lat/lon
        forecast_url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
        
        # Met Norway requires a user agent with contact information
        headers = {
            'User-Agent': 'NorwayTripPlanner/1.0 github.com/ichscheine/air-loc'
        }
        
        # Make the API request
        response = requests.get(forecast_url, headers=headers)
        
        # Debug info
        print(f"Met Norway API response for {lat},{lon}: Status {response.status_code}")
        
        if response.status_code == 200:
            # Process the Met Norway data into a format compatible with our existing UI
            data = response.json()
            processed_data = process_met_norway_data(data)
            return processed_data
        else:
            print(f"Met Norway API error: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting forecast from Met Norway: {e}")
        return None

def process_met_norway_data(data):
    """
    Process Met Norway API data into a format compatible with our existing UI
    """
    try:
        # Create a structure similar to OpenWeatherMap for compatibility
        processed_data = {
            'daily': [],
            'hourly': []
        }
        
        if 'properties' not in data or 'timeseries' not in data['properties']:
            print("Invalid Met Norway data format")
            return None
        
        # Get the timeseries data
        timeseries = data['properties']['timeseries']
        
        # Current date to track day changes
        current_date = None
        daily_data = None
        
        # Process hourly data first (for the next 24 hours)
        for i, time_data in enumerate(timeseries[:24]):  # First 24 hours
            timestamp = datetime.fromisoformat(time_data['time'].replace('Z', '+00:00'))
            
            # Skip if we don't have the necessary data
            if 'data' not in time_data or 'instant' not in time_data['data'] or 'details' not in time_data['data']['instant']:
                continue
                
            details = time_data['data']['instant']['details']
            
            # Find weather symbol for this time (in next_1_hours, next_6_hours, or next_12_hours)
            symbol_code = None
            precipitation_amount = 0
            
            for period in ['next_1_hours', 'next_6_hours', 'next_12_hours']:
                if period in time_data['data'] and 'summary' in time_data['data'][period]:
                    symbol_code = time_data['data'][period]['summary'].get('symbol_code')
                    if 'details' in time_data['data'][period]:
                        precipitation_amount = time_data['data'][period]['details'].get('precipitation_amount', 0)
                    break
            
            # If we still don't have a symbol, use a default
            if not symbol_code:
                symbol_code = 'fair_day'
            
            # Create an hourly entry
            hourly_entry = {
                'dt': int(timestamp.timestamp()),
                'temp': details.get('air_temperature', 0),
                'weather': [{
                    'description': get_weather_description(symbol_code),
                    'icon': convert_met_norway_icon(symbol_code)
                }],
                'wind_speed': details.get('wind_speed', 0),
                'wind_deg': details.get('wind_from_direction', 0),
                'humidity': details.get('relative_humidity', 0),
                'pop': 1.0 if precipitation_amount > 0 else 0.0
            }
            
            processed_data['hourly'].append(hourly_entry)
        
        # Process daily data - group by day
        day_data = {}
        
        for time_data in timeseries:
            timestamp = datetime.fromisoformat(time_data['time'].replace('Z', '+00:00'))
            day_key = timestamp.date().isoformat()
            
            if day_key not in day_data:
                day_data[day_key] = {
                    'temps': [],
                    'weather_symbols': [],
                    'precipitation': [],
                    'wind_speeds': [],
                    'timestamp': int(timestamp.replace(hour=12).timestamp())  # Use noon as representative time
                }
            
            if 'data' in time_data and 'instant' in time_data['data'] and 'details' in time_data['data']['instant']:
                details = time_data['data']['instant']['details']
                
                # Collect temperature for min/max calculation
                if 'air_temperature' in details:
                    day_data[day_key]['temps'].append(details['air_temperature'])
                
                # Collect wind speed
                if 'wind_speed' in details:
                    day_data[day_key]['wind_speeds'].append(details['wind_speed'])
                
                # Find weather symbol and precipitation
                for period in ['next_1_hours', 'next_6_hours', 'next_12_hours']:
                    if period in time_data['data'] and 'summary' in time_data['data'][period]:
                        symbol_code = time_data['data'][period]['summary'].get('symbol_code')
                        if symbol_code:
                            day_data[day_key]['weather_symbols'].append(symbol_code)
                        
                        if 'details' in time_data['data'][period] and 'precipitation_amount' in time_data['data'][period]['details']:
                            day_data[day_key]['precipitation'].append(time_data['data'][period]['details']['precipitation_amount'])
        
        # Convert daily data to the format expected by our UI
        for day_key, data in day_data.items():
            if len(data['temps']) == 0:
                continue  # Skip days with no temperature data
                
            # Get the most common weather symbol for the day
            weather_symbol = max(set(data['weather_symbols']), key=data['weather_symbols'].count) if data['weather_symbols'] else 'fair_day'
            
            # Calculate precipitation probability
            has_precipitation = any(p > 0 for p in data['precipitation']) if data['precipitation'] else False
            
            daily_entry = {
                'dt': data['timestamp'],
                'temp': {
                    'min': min(data['temps']),
                    'max': max(data['temps'])
                },
                'weather': [{
                    'description': get_weather_description(weather_symbol),
                    'icon': convert_met_norway_icon(weather_symbol)
                }],
                'wind_speed': max(data['wind_speeds']) if data['wind_speeds'] else 0,
                'pop': 1.0 if has_precipitation else 0.0
            }
            
            processed_data['daily'].append(daily_entry)
        
        # Sort daily data by timestamp
        processed_data['daily'].sort(key=lambda x: x['dt'])
        
        # Limit to 5 days
        processed_data['daily'] = processed_data['daily'][:5]
        
        return processed_data
        
    except Exception as e:
        print(f"Error processing Met Norway data: {e}")
        return None

def convert_met_norway_icon(symbol_code):
    """
    Convert Met Norway symbol codes to OpenWeatherMap icon codes
    """
    # Remove any '_day' or '_night' suffix for simplicity
    base_code = symbol_code.split('_')[0]
    
    # Map Met Norway symbols to OpenWeatherMap icons
    met_to_owm = {
        'clearsky': '01d',
        'fair': '02d',
        'partlycloudy': '03d',
        'cloudy': '04d',
        'rainshowers': '09d',
        'rain': '10d',
        'heavyrain': '10d',
        'rainandthunder': '11d',
        'sleet': '13d',
        'snow': '13d',
        'snowandthunder': '13d',
        'fog': '50d',
        'sleetshowers': '09d',
        'snowshowers': '13d',
        'rainshowersandthunder': '11d',
        'sleetshowersandthunder': '11d',
        'snowshowersandthunder': '13d',
        'heavyrainandthunder': '11d',
        'heavysleetandthunder': '11d',
        'heavysnowandthunder': '13d',
        'heavyrainshowersandthunder': '11d',
        'heavysleetshowersandthunder': '11d',
        'heavysnowshowersandthunder': '13d'
    }
    
    # Check if it's a night icon
    is_night = '_night' in symbol_code
    
    # Get the matching icon, defaulting to clear sky
    owm_icon = met_to_owm.get(base_code, '01d')
    
    # Change 'd' to 'n' for night icons
    if is_night:
        owm_icon = owm_icon[:-1] + 'n'
    
    return owm_icon

def get_weather_description(symbol_code):
    """
    Get a human-readable description from Met Norway symbol code
    """
    # Remove day/night suffix
    base_code = symbol_code.split('_')[0]
    
    # Map to human-readable descriptions
    descriptions = {
        'clearsky': 'Clear sky',
        'fair': 'Fair',
        'partlycloudy': 'Partly cloudy',
        'cloudy': 'Cloudy',
        'rainshowers': 'Rain showers',
        'rain': 'Rain',
        'heavyrain': 'Heavy rain',
        'rainandthunder': 'Rain and thunder',
        'sleet': 'Sleet',
        'snow': 'Snow',
        'snowandthunder': 'Snow and thunder',
        'fog': 'Fog',
        'sleetshowers': 'Sleet showers',
        'snowshowers': 'Snow showers',
        'rainshowersandthunder': 'Rain showers and thunder',
        'sleetshowersandthunder': 'Sleet showers and thunder',
        'snowshowersandthunder': 'Snow showers and thunder',
        'heavyrainandthunder': 'Heavy rain and thunder',
        'heavysleetandthunder': 'Heavy sleet and thunder',
        'heavysnowandthunder': 'Heavy snow and thunder',
        'heavyrainshowersandthunder': 'Heavy rain showers and thunder',
        'heavysleetshowersandthunder': 'Heavy sleet showers and thunder',
        'heavysnowshowersandthunder': 'Heavy snow showers and thunder'
    }
    
    return descriptions.get(base_code, 'Unknown')

def get_weather_icon(icon_code):
    """
    Map OpenWeatherMap icon codes to emoji for better display
    """
    icons = {
        "01d": "☀️",  # clear sky day
        "01n": "🌙",  # clear sky night
        "02d": "⛅",  # few clouds day
        "02n": "☁️",  # few clouds night
        "03d": "☁️",  # scattered clouds
        "03n": "☁️",
        "04d": "☁️",  # broken clouds
        "04n": "☁️",
        "09d": "🌧️",  # shower rain
        "09n": "🌧️",
        "10d": "🌦️",  # rain day
        "10n": "🌧️",  # rain night
        "11d": "⛈️",  # thunderstorm
        "11n": "⛈️",
        "13d": "❄️",  # snow
        "13n": "❄️",
        "50d": "🌫️",  # mist
        "50n": "🌫️"
    }
    return icons.get(icon_code, "🌡️")

def load_high_quality_image(image_path):
    """
    Load image from path with highest possible quality
    """
    try:
        # Open image with PIL to preserve quality
        img = Image.open(image_path)
        # Convert to bytes to pass directly to streamlit without compression
        buf = io.BytesIO()
        img.save(buf, format="PNG", quality=100)
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return image_path  # Fall back to regular path if there's an error

# ------------------------------
# DATA STRUCTURE FOR YOUR TRIP
# ------------------------------

# Ensure gallery directory exists
gallery_path = Path('Norway_gallery')
gallery_path.mkdir(exist_ok=True)

# Get list of images from Norway_gallery folder
gallery_images = [str(gallery_path / img) for img in os.listdir(gallery_path) if img.endswith(('.png', '.jpg', '.jpeg')) and not img.startswith('.')]
gallery_images.sort()  # Sort images to ensure consistent order

# Map day dates to the proper day key in our mapping
date_to_day_key = {
    "2025-08-02 (Saturday)": "2025-08-02",
    "2025-08-03 (Sunday)": "2025-08-03",
    "2025-08-04 (Monday)": "2025-08-04",
    "2025-08-05 (Tuesday)": "2025-08-05",
    "2025-08-06 (Wednesday)": "2025-08-06",
    "2025-08-07 (Thursday)": "2025-08-07",
    "2025-08-08 (Friday)": "2025-08-08",
    "2025-08-09 (Saturday)": "2025-08-09",
    "2025-08-10 (Sunday)": "2025-08-10",
    "2025-08-11 (Monday)": "2025-08-11",
    "2025-08-12 (Tuesday)": "2025-08-12",
    "2025-08-13 (Wednesday)": "2025-08-13",
    "2025-08-14 (Thursday)": "2025-08-14",
    "2025-08-15 (Friday)": "2025-08-15",
    "2025-08-16 (Saturday)": "2025-08-16",
}

# Define the trip data
trip_data = [
    {
        "date": "2025-08-02 (Saturday)",
        "location": "IAD → FRA → EVE",
        "details": "Overnight flight from Washington Dulles to Evenes via Frankfurt. Departure 6:10 PM.",
        "activities": ["Airport lounges", "In-flight entertainment", "Duty-free shopping"],
        "dining_options": ["In-flight meals", "Airport restaurants (IAD/FRA)", "Airline snacks"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-03 (Sunday)",
        "location": "Evenes → Vågan, Lofoten",
        "details": """Arrive at Evenes Airport at 2:10 PM. Pick up car at Hertz at 3 PM.
Drive ~3 hours to Vågan in Lofoten.
        
**Suggestions en route:**  
- Tjeldsund Bridge photo stop  
- Scenic fjord views along E10
""",
        "activities": ["Scenic driving", "Photography at Tjeldsund Bridge", "Nature walks"],
        "dining_options": ["Evenes Airport cafes", "Roadside stops", "Local restaurants in Lofoten"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-04 (Monday)",
        "location": "Haukland Beach – Uttakleiv Beach – Offersøykammen Hike",
        "details": """- Haukland Beach: white sands, turquoise water  
- Uttakleiv Beach: boulders and sunsets  
- Offersøykammen Hike: panoramic summit views
""",
        "activities": ["Swimming at Haukland Beach", "Sunset photography", "Hiking", "Beachcombing"],
        "dining_options": ["Picnic lunch at the beach", "Cafe in Uttakleiv", "Dinner in Leknes"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-05 (Tuesday)",
        "location": "Steffenakken – Reinebringen – Hamnøy – Ramberg",
        "details": """- Steffenakken Hike: lesser-known viewpoint  
- Reinebringen: iconic ridge above Reine  
- Hamnøy: red rorbuer cabins  
- Ramberg Beach: scenic mountain backdrop
""",
        "activities": ["Hiking Steffenakken", "Photography at Reinebringen", "Exploring fishing villages", "Beach relaxation"],
        "dining_options": ["Breakfast in Leknes", "Anita's Sjømat in Hamnøy", "Dinner at Ramberg Gjestegård"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-06 (Wednesday)",
        "location": "Nusfjord?",
        "details": """Visit traditional fishing village Nusfjord.

**Other ideas:**  
- Lofoten Museum  
- Sea eagle safari
""",
        "activities": ["Exploring Nusfjord village", "Sea eagle safari", "Museum visit", "Cod liver oil factory tour"],
        "dining_options": ["Karoline Restaurant in Nusfjord", "Local seafood at Nusfjord Rorbuer", "Traditional stockfish meal"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-07 (Thursday)",
        "location": "Henningsvær + Fløya",
        "details": """- Henningsvær: charming harbor, art galleries, football field  
- Fløya hike: spectacular harbor views
""",
        "activities": ["Art gallery tours", "Visit famous floating football field", "Hiking Fløya", "Shopping for local crafts"],
        "dining_options": ["Fiskekrogen seafood restaurant", "Henningsvær Lysstøperi & Café", "Trevarefabrikken brewery"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-08 (Friday)",
        "location": "Fly EVE → Bergen",
        "details": "Flight 3:40 PM – 5:30 PM to Bergen.",
        "activities": ["Souvenir shopping in Lofoten", "Flight to Bergen", "Evening walk in Bergen"],
        "dining_options": ["Airport dining", "In-flight meal", "Dinner at Bryggen in Bergen"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-09 (Saturday)",
        "location": "Bergen → Ålesund",
        "details": """Morning in Bergen:
- Bryggen Wharf
- Fløyen funicular
- Fish market

Fly to Ålesund in afternoon.
""",
        "activities": ["Bryggen Wharf exploration", "Fløyen funicular ride", "Fish market visit", "Flight to Ålesund"],
        "dining_options": ["Bergen fish market food stalls", "Bryggen restaurants", "Evening dining in Ålesund"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-10 (Sunday)",
        "location": "Drive to Geirangerfjorden",
        "details": """Scenic drive to Geirangerfjord.

Possible hikes:  
- Skageflå Farm hike  
- Flydalsjuvet viewpoint
""",
        "activities": ["Scenic coastal drive", "Skageflå Farm hike", "Flydalsjuvet viewpoint visit", "Photography stops"],
        "dining_options": ["Packed lunch for the drive", "Local farm-to-table restaurants", "Dinner at Geiranger hotel"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-11 (Monday)",
        "location": "Geirangerfjord Cruise",
        "details": """- Seven Sisters waterfall  
- The Suitor waterfall  
- UNESCO fjord scenery
""",
        "activities": ["Geirangerfjord cruise", "Seven Sisters waterfall viewing", "The Suitor waterfall visit", "UNESCO fjord scenery exploration"],
        "dining_options": ["Breakfast at hotel", "Lunch on cruise ship", "Dinner at local seafood restaurant"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-12 (Tuesday)",
        "location": "Geirangerfjord → Ålesund",
        "details": "Morning canyoning in Geirangerfjord. Drive back to Ålesund.",
        "activities": ["Morning canyoning adventure", "Scenic drive to Ålesund", "Evening walk in Ålesund", "Art nouveau architecture exploration"],
        "dining_options": ["Breakfast at Geiranger", "Lunch at roadside café", "Dinner at Ålesund seafood restaurant"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-13 (Wednesday)",
        "location": "Fly Ålesund → Stavanger",
        "details": """- Explore Stavanger Old Town  
- Colorful wooden houses and harbor views
""",
        "activities": ["Morning in Ålesund", "Flight to Stavanger", "Stavanger Old Town exploration", "Harbor views walking tour"],
        "dining_options": ["Breakfast at Ålesund hotel", "Airport lunch", "Dinner in Stavanger Old Town"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-14 (Thursday)",
        "location": "Kjerag Hike",
        "details": """- Kjeragbolten hike: iconic boulder wedged between cliffs
""",
        "activities": ["Kjeragbolten full day hike", "Photography at the iconic boulder", "Scenic views of Lysefjord"],
        "dining_options": ["Packed lunch for hike", "Early breakfast at hotel", "Celebratory dinner in Stavanger"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-15 (Friday)",
        "location": "Pulpit Rock",
        "details": """- Hike to Preikestolen (Pulpit Rock) overlooking Lysefjord
""",
        "activities": ["Hike to Preikestolen (Pulpit Rock)", "Photography at the cliff edge", "Lysefjord views", "Nature exploration"],
        "dining_options": ["Breakfast at accommodation", "Packed lunch for hike", "Dinner at local restaurant in Stavanger"],
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-16 (Saturday)",
        "location": "Return Home",
        "details": "Flight SVG → FRA → IAD. Depart 6:45 AM, arrive 1:30 PM.",
        "activities": ["Early morning airport transfer", "SVG → FRA flight", "FRA → IAD flight", "Arrival home"],
        "dining_options": ["Early breakfast at hotel", "In-flight meals", "Airport dining options"],
        "images": [],  # Will be populated after full trip_data is defined
    }
]

# Norway locations with Google Maps links for downloading images
norway_locations = {
    "2025-08-02 (Saturday)": ["https://www.google.com/maps/place/Evenes+Airport/@68.4891814,16.6695702,14z/"],
    "2025-08-03 (Sunday)": ["https://www.google.com/maps/place/Lofoten/@68.2217788,13.3457272,9z/", 
                          "https://www.google.com/maps/place/Tjeldsund+Bridge/@68.6268815,16.575456,15z/"],
    "2025-08-04 (Monday)": ["https://www.google.com/maps/place/Uttakleiv+beach/@68.2097322,13.5050698,15z/",
                          "https://www.google.com/maps/place/Haukland+Beach/@68.1924842,13.5156587,15z/"],
    "2025-08-05 (Tuesday)": ["https://www.google.com/maps/place/Reinebringen/@67.9319532,13.0787456,15z/",
                           "https://www.google.com/maps/place/Hamnøy/@67.9394111,13.1369071,15z/"],
    "2025-08-06 (Wednesday)": ["https://www.google.com/maps/place/Nusfjord/@68.0352924,13.3491096,15z/"],
    "2025-08-07 (Thursday)": ["https://www.google.com/maps/place/Henningsvær/@68.1483917,14.2015219,14z/",
                             "https://www.google.com/maps/place/Fløya/@68.2187728,14.5661099,14z/"],
    "2025-08-08 (Friday)": ["https://www.google.com/maps/place/Bergen/@60.3943036,5.3259192,12z/"],
    "2025-08-09 (Saturday)": ["https://www.google.com/maps/place/Bryggen/@60.3973883,5.3233519,17z/",
                             "https://www.google.com/maps/place/Mount+Fløyen/@60.3953049,5.341528,15z/"],
    "2025-08-10 (Sunday)": ["https://www.google.com/maps/place/Geiranger+Fjord/@62.1048487,7.0752131,12z/",
                           "https://www.google.com/maps/place/Flydalsjuvet/@62.0891863,7.2232614,15z/"],
    "2025-08-11 (Monday)": ["https://www.google.com/maps/place/Seven+Sisters+waterfall/@62.1163866,7.0870177,15z/"],
    "2025-08-12 (Tuesday)": ["https://www.google.com/maps/place/Ålesund/@62.4722284,6.1524231,12z/"],
    "2025-08-13 (Wednesday)": ["https://www.google.com/maps/place/Old+Stavanger/@58.9738607,5.7313304,17z/"],
    "2025-08-14 (Thursday)": ["https://www.google.com/maps/place/Kjeragbolten/@59.0349826,6.5875256,15z/"],
    "2025-08-15 (Friday)": ["https://www.google.com/maps/place/Pulpit+Rock/@58.9861151,6.18994,15z/"],
    "2025-08-16 (Saturday)": ["https://www.google.com/maps/place/Stavanger+Airport,+Sola/@58.8818966,5.6264149,14z/"]
}

# Check if we have an image mapping file from our fetcher
mapping_file = gallery_path / "image_mapping.json"
day_to_images = {}

# Function to update image mapping based on filenames
def update_image_mapping():
    global day_to_images
    mapping = {date_to_day_key[date]: [] for date in date_to_day_key}
    
    # Create a dictionary of keywords for each day
    day_keywords = {}
    for day in trip_data:
        day_key = date_to_day_key.get(day["date"])
        if day_key:
            # Get keywords from the location name
            location = day["location"].lower()
            keywords = re.sub(r'[^\w\s]', ' ', location).split()
            # Only keep keywords with 3+ characters
            keywords = [k for k in keywords if len(k) >= 3]
            day_keywords[day_key] = keywords
    
    # Assign images to days based on filename keywords
    for img_path in gallery_images:
        img_name = os.path.basename(img_path).lower()
        assigned = False
        
        # First, try to match by day number in the filename (e.g., "2025-08-04" or just "04")
        for day_key in mapping:
            day_number = day_key.split('-')[-1]  # Get "04" from "2025-08-04"
            if day_number in img_name or day_key in img_name:
                mapping[day_key].append(img_path)
                assigned = True
                break
        
        # If not assigned by day number, try to match by location keywords
        if not assigned:
            for day_key, keywords in day_keywords.items():
                if any(keyword in img_name for keyword in keywords):
                    mapping[day_key].append(img_path)
                    assigned = True
                    break
    
    # For any unassigned images, we'll leave them out of the mapping
    # but they'll still be accessible through the fallback mechanism
    
    return mapping

if mapping_file.exists():
    try:
        with open(mapping_file, 'r') as f:
            day_to_images = json.load(f)
        print(f"Loaded image mapping from {mapping_file}")
        
        # Check if we need to update the mapping (if there are new images)
        all_mapped_images = []
        for images in day_to_images.values():
            all_mapped_images.extend(images)
        
        if len(gallery_images) > len(all_mapped_images):
            print("Found new images, updating mapping...")
            day_to_images = update_image_mapping()
            
            # Save the updated mapping
            with open(mapping_file, 'w') as f:
                json.dump(day_to_images, f, indent=2)
            print("Updated image mapping saved.")
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        # Create a new mapping
        day_to_images = update_image_mapping()
else:
    # Create a new mapping
    day_to_images = update_image_mapping()
    
    # Save the mapping
    with open(mapping_file, 'w') as f:
        json.dump(day_to_images, f, indent=2)
    print(f"Created new image mapping at {mapping_file}")

# Function to get images for a specific day
def get_day_images(day_date):
    # First try to get images from our mapping file
    day_key = date_to_day_key.get(day_date)
    if day_key and day_key in day_to_images and day_to_images[day_key]:
        return day_to_images[day_key]
    
    # If no mapping found, try to find images by name matching
    if gallery_images:
        # Get location name for this day
        location_name = ""
        for day in trip_data:
            if day["date"] == day_date:
                location_name = day["location"].lower()
                break
        
        # Convert location to keywords by splitting and removing special characters
        keywords = re.sub(r'[^\w\s]', ' ', location_name).split()
        
        # Find any images that match keywords in the filename
        matching_images = []
        for img_path in gallery_images:
            img_name = os.path.basename(img_path).lower()
            if any(keyword in img_name for keyword in keywords if len(keyword) > 2):
                matching_images.append(img_path)
        
        if matching_images:
            return matching_images
    
    # If no images match, return empty list (don't use fallback)
    return []

# Now populate the images for each day using our mapping function
for day in trip_data:
    day["images"] = get_day_images(day["date"])

# ------------------------------
# STREAMLIT PAGE
# ------------------------------

st.set_page_config(page_title="Norway Adventure 2025", layout="wide")

# Load custom CSS
with open('style.css') as f:
    css_content = f.read()

# Add additional CSS for high-quality images and weather display
css_content += """
/* Enhanced Typography and Visual Hierarchy */
.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-align: center;
}

.sidebar-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 16px;
    margin: -1rem -1rem 1rem -1rem;
    font-weight: 600;
    font-size: 1.1rem;
    border-radius: 8px 8px 0 0;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.location-header {
    background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.date-subheader {
    text-align: center;
    color: #666;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 1rem;
    font-size: 1.1rem;
}

.section-header {
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 2rem 0 1rem 0;
    font-weight: 600;
}

/* High-quality image rendering enhancements */
.high-quality-image img {
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
    -webkit-backface-visibility: hidden;
    -ms-interpolation-mode: bicubic;
    transform: translateZ(0);
}

/* Enhanced Content Cards */
.content-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(66, 133, 244, 0.1);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.content-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4285F4, #34A853, #FBBC04, #EA4335);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.content-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    border-color: rgba(66, 133, 244, 0.2);
}

.content-card:hover::before {
    opacity: 1;
}

.content-card h4 {
    margin-top: 0;
    margin-bottom: 15px;
    color: #1a202c;
    font-weight: 600;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Weather display styling */
.weather-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
}

.weather-card h4 {
    color: white;
    margin-bottom: 15px;
}

.weather-card .card-content {
    color: white;
}

.forecast-day {
    background-color: #fff;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 10px;
    height: 100%;
    transition: transform 0.2s;
}

.forecast-day:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.hourly-forecast-item {
    transition: transform 0.2s;
}

.hourly-forecast-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

/* Add styles for Google Maps links and ratings */
.place-rating {
    display: flex;
    align-items: center;
    justify-content: flex-end;
}

.map-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background-color: #4285F4;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    margin-left: 8px;
    text-decoration: none;
    font-size: 12px;
    transition: all 0.3s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.map-link:hover {
    background-color: #3367D6;
    transform: scale(1.1);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

.rating-stars {
    color: #FFD700;
    margin-right: 4px;
}

/* Add styles for interactive elements */
a {
    text-decoration: none;
    position: relative;
    color: #4285F4;
    transition: all 0.2s;
}

a:hover {
    text-decoration: underline;
    color: #3367D6;
}

a.interactive-link {
    border-bottom: 1px dashed #4285F4;
}

a.interactive-link:hover {
    border-bottom: 1px solid #3367D6;
    background-color: rgba(66, 133, 244, 0.1);
    border-radius: 2px;
}

.clickable-item {
    transition: all 0.2s;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 2px 4px;
}

.clickable-item:hover {
    border-color: #4285F4;
    background-color: rgba(66, 133, 244, 0.1);
    cursor: pointer;
}

.interactive-card {
    position: relative;
    overflow: hidden;
}

.interactive-card::after {
    content: "";
    position: absolute;
    bottom: 0;
    right: 0;
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 0 0 20px 20px;
    border-color: transparent transparent rgba(66, 133, 244, 0.5) transparent;
    opacity: 0;
    transition: opacity 0.3s;
}

.interactive-card:hover::after {
    opacity: 1;
}

/* Enhanced Navigation */
.day-nav-buttons {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(66, 133, 244, 0.1);
}

/* Enhanced Progress Indicator */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4285F4, #34A853, #FBBC04, #EA4335) !important;
    height: 8px !important;
    border-radius: 4px !important;
}

/* Enhanced Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    border: 1px solid transparent !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4285F4 0%, #34A853 100%) !important;
    border: none !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
    border: 1px solid rgba(66, 133, 244, 0.2) !important;
    color: #4285F4 !important;
}

/* Enhanced Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
}

/* Enhanced Selectbox Styling */
.stSelectbox > div > div {
    background-color: white !important;
    border: 1px solid rgba(66, 133, 244, 0.2) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div:focus-within {
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2) !important;
}

/* Day status indicators styling */
.day-status-indicator {
    background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%);
    padding: 8px;
    border-radius: 6px;
    margin: 10px 0;
    text-align: center;
    font-size: 0.8rem;
    border: 1px solid rgba(76, 175, 80, 0.2);
}

/* Enhanced Checklist Categories */
.checklist-category {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 12px;
    margin: 10px -10px 8px -10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.9rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.critical_essentials-category {
    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%) !important;
}

.weather___safety__norway_essential_-category {
    background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%) !important;
}

.core_clothing-category {
    background: linear-gradient(135deg, #45B7D1 0%, #96CEB4 100%) !important;
}

.hiking___outdoor_gear-category {
    background: linear-gradient(135deg, #F39C12 0%, #D68910 100%) !important;
}

.health___personal_care-category {
    background: linear-gradient(135deg, #9B59B6 0%, #8E44AD 100%) !important;
}

.essential_tech-category {
    background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%) !important;
}

.comfort___convenience-category {
    background: linear-gradient(135deg, #16A085 0%, #138D75 100%) !important;
}

.norway_specific_extras-category {
    background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%) !important;
}

.entertainment___optional-category {
    background: linear-gradient(135deg, #F1C40F 0%, #D4AC0D 100%) !important;
}

/* Enhanced Activity and Dining Items */
.activity-item, .dining-item {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(66, 133, 244, 0.1);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.activity-item:hover, .dining-item:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(66, 133, 244, 0.15);
    border-color: rgba(66, 133, 244, 0.3);
    background: linear-gradient(145deg, #f0f8ff 0%, #e6f3ff 100%);
}

/* Enhanced Image Container */
.image-container {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    margin-bottom: 15px;
}

.image-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
}

.image-container img {
    border-radius: 12px;
    transition: transform 0.3s ease;
}

.image-container:hover img {
    transform: scale(1.02);
}

/* 3-day forecast styling */
.forecast-day-compact {
    background-color: #fff;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 5px;
    text-align: center;
    flex: 1;
}

/* Styling for location links in the sidebar */
.location-link-box {
    display: flex;
    align-items: center;
    background-color: #fff;
    border-radius: 8px;
    padding: 10px;
    transition: all 0.2s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid transparent;
    margin-bottom: 6px;
}

.location-link-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    background-color: #f0f8ff;
    border-color: #4285F4;
    cursor: pointer;
}

.location-link-number {
    background-color: #4285F4;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: bold;
    margin-right: 12px;
    flex-shrink: 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    transition: all 0.2s;
}

.location-link-box:hover .location-link-number {
    background-color: #3367D6;
    transform: scale(1.1);
    box-shadow: 0 3px 6px rgba(0,0,0,0.3);
}

.location-link-name {
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s;
}

.location-link-box:hover .location-link-name {
    color: #4285F4;
    font-weight: 600;
}

/* Add responsive design for forecast and location links on mobile */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem !important;
    }
    
    .content-card {
        padding: 15px !important;
        margin-bottom: 15px !important;
    }
    
    .day-nav-buttons {
        padding: 15px !important;
    }
    
    .location-header {
        font-size: 1.3rem !important;
    }
    
    .activity-item, .dining-item {
        padding: 8px !important;
    }
    
    .activity-icon, .dining-icon {
        font-size: 1rem !important;
    }
    
    .forecast-day {
        margin-bottom: 8px;
        padding: 8px;
    }
    
    .forecast-day-compact {
        padding: 5px;
        margin-bottom: 5px;
    }
    
    .location-link-box {
        padding: 6px;
        margin-bottom: 6px;
    }
    
    .location-link-number {
        width: 20px;
        height: 20px;
        font-size: 0.8rem;
    }
    
    .location-link-name {
        font-size: 0.8rem;
    }
}

/* Loading states and animations */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Enhanced tooltips */
[title] {
    position: relative;
}

/* Print styles */
@media print {
    .stSidebar {
        display: none !important;
    }
    
    .day-nav-buttons {
        display: none !important;
    }
    
    .content-card {
        break-inside: avoid;
        box-shadow: none !important;
        border: 1px solid #ccc !important;
    }
}
"""

st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

# Define the checklist data
checklist = {
    "Critical Essentials": [
        "passport",
        "flight tickets and confirmations",
        "travel insurance documents",
        "prescription medications",
        "credit cards and some cash (NOK/EUR)",
        "phone, charger, and adapter (EU), powerbank",
        "U.S. driver license",
        "hotel/accommodation confirmations",
        "car rental confirmation"
    ],
    "Weather & Safety (Norway Essential)": [
        "waterproof jacket/rain coat",
        "sturdy hiking boots (broken in)",
        "warm layers (August can be cool, 10-20°C)",
        "sunglasses (essential for glacial areas)",
        "sunscreen (SPF 30+, UV is strong)",
        "first aid kit basics",
        "waterproof gear (frequent rain)",
        "thermal underwear (for cool evenings)",
        "warm hat/beanie"
    ],
    "Core Clothing": [
        "layered clothing (temperatures vary)",
        "comfortable walking pants",
        "quick-dry hiking pants",
        "warm sweater or fleece",
        "hiking socks (wool or synthetic)",
        "underwear and socks (extra pairs)",
        "comfortable casual clothes",
        "sleepwear"
    ],
    "Hiking & Outdoor Gear": [
        "day backpack (20-30L)",
        "good grip shoes (wet rocks)",
        "waterproof pants (for boat tours)",
        "hiking gloves",
        "quick-dry base layers",
        "trekking poles (optional)",
        "dry bags (keep electronics dry)"
    ],
    "Health & Personal Care": [
        "motion sickness relief",
        "insect/mosquito repellent",
        "pain relievers",
        "personal hygiene items",
        "lip balm with SPF",
        "contact lenses/glasses (backup)",
        "hand sanitizer"
    ],
    "Essential Tech": [
        "camera and extra batteries/memory cards",
        "waterproof phone case",
        "portable charger/power bank",
        "travel adapter (Type C/F for Norway)",
        "downloaded maps (offline access)",
        "headphones"
    ],
    "Comfort & Convenience": [
        "comfortable walking shoes (non-hiking)",
        "water bottle (reusable)",
        "umbrella or poncho",
        "warm scarf or buff",
        "eye mask and earplugs",
        "slippers for accommodation",
        "travel pillow",
        "snacks for hiking",
        "small daypack for excursions"
    ],
    "Norway Specific Extras": [
        "cash (some places don't take cards)",
        "binoculars (for wildlife viewing)",
        "swim clothes (for hot springs/beaches)"
    ],
    "Entertainment & Optional": [
        "selfie stick",
        "Kindle/books"
    ]
}

# Create the main header with a more compact style
st.markdown('<div class="main-header"><h1>🇳🇴 Norway Adventure - August 2025</h1></div>', unsafe_allow_html=True)

# Initialize session state for selected date
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = trip_data[0]["date"]

# Initialize session state for personal notes
if 'personal_notes' not in st.session_state:
    st.session_state.personal_notes = {}
    
# Initialize session state for favorite places
if 'favorite_places' not in st.session_state:
    st.session_state.favorite_places = set()

# Initialize session state for visited status
if 'visited_status' not in st.session_state:
    st.session_state.visited_status = {}

# Initialize session state for personal preferences
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'preferred_activities': [],
        'dietary_restrictions': [],
        'budget_level': 'medium',
        'fitness_level': 'moderate',
        'weather_preference': 'any'
    }

# First add the packing checklist at the top of the sidebar
st.sidebar.markdown('<div class="sidebar-header">📋 Packing Checklist</div>', unsafe_allow_html=True)

# Add checklist to sidebar
with st.sidebar.expander("View Checklist", expanded=False):
    # Initialize session state for checklist if not exists
    if 'checklist_state' not in st.session_state:
        st.session_state.checklist_state = {}
        for category, items in checklist.items():
            for item in items:
                st.session_state.checklist_state[item] = False
    
    # Display checklist with styling based on categories
    for category, items in checklist.items():
        category_slug = category.lower().replace("/", "-").replace(" ", "_")
        st.markdown(f'<div class="checklist-category {category_slug}-category">{category}</div>', unsafe_allow_html=True)
        for i, item in enumerate(items):
            # Create a more unique key by combining category, index and a sanitized version of the item
            item_slug = item.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(":", "").replace(";", "").replace(",", "")
            unique_key = f"check_{category_slug}_{i}_{item_slug[:10]}"
            checked = st.checkbox(item, key=unique_key, value=st.session_state.checklist_state.get(item, False))
            st.session_state.checklist_state[item] = checked

# Add personal preferences panel
st.sidebar.markdown('<div class="sidebar-header">⚙️ Personal Preferences</div>', unsafe_allow_html=True)

with st.sidebar.expander("Customize Your Trip", expanded=True):
    st.markdown("**👇 Select your preferences to get personalized recommendations!**")
    
    st.markdown("**Activity Preferences:**")
    activity_options = ["Hiking", "Photography", "Museums", "Beaches", "Scenic Drives", "Food Tours", "Wildlife Viewing", "Adventure Sports"]
    selected_activities = st.multiselect(
        "What activities interest you most?",
        activity_options,
        default=st.session_state.preferences['preferred_activities'],
        key="activity_prefs",
        help="Select activities you enjoy - you'll get personalized recommendations!"
    )
    st.session_state.preferences['preferred_activities'] = selected_activities
    
    st.markdown("**Fitness Level:**")
    fitness_level = st.radio(
        "How would you rate your fitness level?",
        ["Light", "Moderate", "Active", "Very Active"],
        index=["light", "moderate", "active", "very_active"].index(st.session_state.preferences['fitness_level']) if st.session_state.preferences['fitness_level'] in ["light", "moderate", "active", "very_active"] else 1,
        key="fitness_level",
        help="This affects hiking and activity recommendations"
    )
    st.session_state.preferences['fitness_level'] = fitness_level.lower()
    
    st.markdown("**Budget Level:**")
    budget_level = st.select_slider(
        "What's your budget preference?",
        options=["Budget", "Medium", "Premium", "Luxury"],
        value=st.session_state.preferences['budget_level'].title(),
        key="budget_level",
        help="Get dining and activity suggestions matching your budget"
    )
    st.session_state.preferences['budget_level'] = budget_level.lower()
    
    st.markdown("**Dietary Restrictions:**")
    dietary_options = ["None", "Vegetarian", "Vegan", "Gluten-free", "Seafood allergy", "Nut allergy"]
    dietary_restrictions = st.multiselect(
        "Any dietary restrictions?",
        dietary_options,
        default=st.session_state.preferences['dietary_restrictions'],
        key="dietary_restrictions",
        help="Get personalized dining recommendations and warnings"
    )
    st.session_state.preferences['dietary_restrictions'] = dietary_restrictions
    
    # Show what will change based on selections
    if selected_activities or fitness_level != "Moderate" or budget_level != "Medium" or (dietary_restrictions and "None" not in dietary_restrictions):
        st.success("✅ Preferences saved! Look for:")
        changes = []
        if selected_activities:
            changes.append("🎯 Personalized activity recommendations")
        if budget_level != "medium":
            changes.append("💰 Budget-specific dining suggestions") 
        if dietary_restrictions and "None" not in dietary_restrictions:
            changes.append("🍽️ Dietary restriction alerts")
        if fitness_level in ["light", "very_active"]:
            changes.append("🥾 Fitness-matched activity advice")
            
        for change in changes:
            st.write(f"• {change}")
        st.info("💡 Navigate to different days to see personalized recommendations appear!")
    else:
        st.info("💡 Make selections above to see personalized recommendations throughout your trip!")

# Add trip progress tracker
st.sidebar.markdown('<div class="sidebar-header">📈 Trip Progress</div>', unsafe_allow_html=True)

with st.sidebar.expander("Track Your Journey", expanded=False):
    # Calculate progress
    total_days = len(trip_data)
    completed_days = len([day for day in trip_data if st.session_state.visited_status.get(day["date"], False)])
    progress_percentage = (completed_days / total_days) * 100 if total_days > 0 else 0
    
    st.metric("Days Completed", f"{completed_days}/{total_days}", f"{progress_percentage:.1f}%")
    
    # Favorite places count
    favorite_count = len(st.session_state.favorite_places)
    st.metric("Favorite Places", favorite_count)
    
    # Notes count
    notes_count = len([note for note in st.session_state.personal_notes.values() if note.strip()])
    st.metric("Personal Notes", notes_count)

# Now add the day selection after the checklist
st.sidebar.markdown('<div class="sidebar-header">📅 Select a Day</div>', unsafe_allow_html=True)

# Create dropdown for day selection
day_options = [f"{day['date']} - {day['location']}" for day in trip_data]
current_selection = None

# Find current selection index
for i, day in enumerate(trip_data):
    if day["date"] == st.session_state.selected_date:
        current_selection = i
        break

if current_selection is None:
    current_selection = 0

selected_index = st.sidebar.selectbox(
    "Choose your day:",
    range(len(day_options)),
    format_func=lambda x: day_options[x],
    index=current_selection,
    key="day_selector"
)

# Update selected date based on dropdown selection
new_selected_date = trip_data[selected_index]["date"]
if new_selected_date != st.session_state.selected_date:
    st.session_state.selected_date = new_selected_date
    st.rerun()

# Add quick navigation buttons for previous/next day
col1, col2 = st.sidebar.columns(2)
with col1:
    if selected_index > 0:
        if st.button("⬅️ Previous", key="prev_day_sidebar", use_container_width=True):
            st.session_state.selected_date = trip_data[selected_index - 1]["date"]
            st.rerun()
with col2:
    if selected_index < len(trip_data) - 1:
        if st.button("Next ➡️", key="next_day_sidebar", use_container_width=True):
            st.session_state.selected_date = trip_data[selected_index + 1]["date"]
            st.rerun()

# Show current day status indicators
current_day = trip_data[selected_index]
status_indicators = []

if st.session_state.visited_status.get(current_day["date"], False):
    status_indicators.append("✅ Visited")
if current_day["location"] in st.session_state.favorite_places:
    status_indicators.append("💖 Favorite")
if st.session_state.personal_notes.get(current_day["date"], "").strip():
    status_indicators.append("📝 Has Notes")

if status_indicators:
    st.sidebar.markdown(
        f'<div style="background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%); '
        f'padding: 8px; border-radius: 6px; margin: 10px 0; text-align: center; font-size: 0.8rem;">'
        f'{" • ".join(status_indicators)}</div>',
        unsafe_allow_html=True
    )

# Hidden functionality for developers - add a small discrete link at the bottom of the sidebar
with st.sidebar.expander("⚙️ Developer Options", expanded=False):
    # Add trip overview dashboard button
    if st.button("📊 Trip Overview Dashboard", key="dashboard_toggle", use_container_width=True):
        st.session_state.show_dashboard = not st.session_state.get('show_dashboard', False)
    
    # Display dashboard content when toggled on
    if st.session_state.get('show_dashboard', False):
        st.markdown('<div class="sidebar-header" style="margin: 20px 0;">📊 Trip Overview Dashboard</div>', unsafe_allow_html=True)
        
        # Trip statistics
        col1, col2 = st.columns(2)
        
        with col1:
            total_days = len(trip_data)
            st.metric("Total Days", total_days, "15-day adventure")
            
            completed_days = len([day for day in trip_data if st.session_state.visited_status.get(day["date"], False)])
            completion_rate = (completed_days / total_days) * 100 if total_days > 0 else 0
            st.metric("Completed", f"{completed_days}/{total_days}", f"{completion_rate:.1f}%")
        
        with col2:
            favorite_count = len(st.session_state.favorite_places)
            st.metric("Favorite Places", favorite_count, "❤️")
            
            notes_count = len([note for note in st.session_state.personal_notes.values() if note.strip()])
            st.metric("Personal Notes", notes_count, "📝")
        
        # Preference matching analysis
        st.markdown("### 🎯 Preference Match")
        
        user_activities = st.session_state.preferences.get('preferred_activities', [])
        if user_activities:
            match_scores = []
            for day in trip_data:
                location_activities = ' '.join(day.get('activities', [])).lower()
                matches = sum(1 for pref in user_activities if pref.lower() in location_activities)
                match_scores.append(matches)
            
            avg_match = sum(match_scores) / len(match_scores) if match_scores else 0
            max_possible = len(user_activities)
            match_percentage = (avg_match / max_possible) * 100 if max_possible > 0 else 0
            
            st.progress(match_percentage / 100)
            st.write(f"**{match_percentage:.1f}%** match")
            
            # Show top matching days
            day_matches = [(trip_data[i], match_scores[i]) for i in range(len(trip_data))]
            day_matches.sort(key=lambda x: x[1], reverse=True)
            
            if day_matches[0][1] > 0:
                st.write("**Best days:**")
                for day, score in day_matches[:2]:
                    if score > 0:
                        st.write(f"• {day['date']} - {day['location']}")
        
        # Budget and fitness analysis
        budget_level = st.session_state.preferences.get('budget_level', 'medium')
        st.write(f"**Budget:** {budget_level.title()}")
        
        fitness_level = st.session_state.preferences.get('fitness_level', 'moderate')
        st.write(f"**Fitness:** {fitness_level.title()}")
        
        # Packing progress
        packed_count = sum(1 for item in st.session_state.checklist_state.values() if item)
        total_items = len([item for category in checklist.values() for item in category])
        packing_progress = (packed_count / total_items) * 100 if total_items > 0 else 0
        
        st.markdown("### 🎒 Packing")
        st.progress(packing_progress / 100)
        st.write(f"**{packed_count}/{total_items}** items ({packing_progress:.1f}%)")
        
        if packing_progress < 100:
            remaining = total_items - packed_count
            days_until_trip = (datetime(2025, 8, 2) - datetime.now()).days
            if days_until_trip > 0:
                items_per_day = remaining / days_until_trip
                st.write(f"💡 Pack {items_per_day:.1f} items/day")
    
    # Add button to refresh image mapping
    if st.button("🔄 Refresh Image Mapping", type="primary", use_container_width=True):
        day_to_images = update_image_mapping()
        
        # Save the updated mapping
        with open(mapping_file, 'w') as f:
            json.dump(day_to_images, f, indent=2)
        
        st.success("Image mapping refreshed!")
        st.rerun()
    
    # Add button to open gallery folder
    if st.button("📁 Open Images Folder", type="secondary", use_container_width=True):
        try:
            # Open the gallery folder with the appropriate command for the OS
            if platform.system() == "Windows":
                os.startfile(str(gallery_path.absolute()))
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", str(gallery_path.absolute())])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(gallery_path.absolute())])
            
            st.success(f"Opened: {gallery_path.absolute()}")
        except Exception as e:
            st.error(f"Could not open folder: {e}")
            st.code(f"Folder path: {gallery_path.absolute()}")
    
    # Show image count
    total_images = len(gallery_images)
    st.write(f"📸 Total images: {total_images}")
    
    # Debug preferences
    if st.checkbox("🔍 Show Preferences Debug"):
        st.write("**Current Preferences:**")
        st.json(st.session_state.preferences)
        st.write("**Personal Notes:**")
        st.json(st.session_state.personal_notes)
        st.write("**Favorite Places:**")
        st.write(list(st.session_state.favorite_places))
        st.write("**Visited Status:**")
        st.json(st.session_state.visited_status)

# Add Export & Sharing section
st.sidebar.markdown('<div class="sidebar-header">📤 Export & Share</div>', unsafe_allow_html=True)

with st.sidebar.expander("Export Your Trip", expanded=False):
    # Export personal notes
    if st.button("📝 Export Notes as Text", use_container_width=True):
        notes_content = "# Norway Trip 2025 - Personal Notes\n\n"
        for date, note in st.session_state.personal_notes.items():
            if note.strip():
                notes_content += f"## {date}\n{note}\n\n"
        
        st.download_button(
            label="Download Notes.txt",
            data=notes_content,
            file_name="norway_trip_notes.txt",
            mime="text/plain"
        )
    
    # Export itinerary summary
    if st.button("📋 Export Itinerary Summary", use_container_width=True):
        itinerary_content = "# Norway Adventure 2025 - Itinerary\n\n"
        for day in trip_data:
            visited_status = "✅ Visited" if st.session_state.visited_status.get(day["date"], False) else "⏳ Planned"
            favorite_status = "💖 Favorite" if day["location"] in st.session_state.favorite_places else ""
            
            itinerary_content += f"## {day['date']} - {day['location']} {visited_status} {favorite_status}\n\n"
            itinerary_content += f"{day['details']}\n\n"
            
            if day.get('activities'):
                itinerary_content += "**Activities:**\n"
                for activity in day['activities']:
                    itinerary_content += f"- {activity}\n"
                itinerary_content += "\n"
            
            if day.get('dining_options'):
                itinerary_content += "**Dining:**\n"
                for dining in day['dining_options']:
                    itinerary_content += f"- {dining}\n"
                itinerary_content += "\n"
            
            # Add personal notes if any
            personal_note = st.session_state.personal_notes.get(day["date"], "")
            if personal_note.strip():
                itinerary_content += f"**Personal Notes:**\n{personal_note}\n\n"
            
            itinerary_content += "---\n\n"
        
        st.download_button(
            label="Download Itinerary.md",
            data=itinerary_content,
            file_name="norway_trip_itinerary.md",
            mime="text/markdown"
        )
    
    # Export packing checklist
    if st.button("🎒 Export Packing List", use_container_width=True):
        packed_items = []
        unpacked_items = []
        
        for category, items in checklist.items():
            for item in items:
                if st.session_state.checklist_state.get(item, False):
                    packed_items.append(f"✅ {item}")
                else:
                    unpacked_items.append(f"⏳ {item}")
        
        checklist_content = "# Norway Trip 2025 - Packing Checklist\n\n"
        checklist_content += f"## Packed Items ({len(packed_items)})\n"
        checklist_content += "\n".join(packed_items) + "\n\n"
        checklist_content += f"## Still to Pack ({len(unpacked_items)})\n"
        checklist_content += "\n".join(unpacked_items)
        
        st.download_button(
            label="Download Packing_List.txt",
            data=checklist_content,
            file_name="norway_packing_checklist.txt",
            mime="text/plain"
        )

selected_date = st.session_state.selected_date

# Show details for selected date
for day in trip_data:
    if day["date"] == selected_date:
        # Create a cleaner navigation with compact controls
        # Find the current day index
        current_index = trip_data.index(day)
        
        # Create navigation with more compact layout
        col1, col2, col3 = st.columns([1, 6, 1])
        
        # Previous day button with improved styling
        if current_index > 0:
            prev_day = trip_data[current_index - 1]
            if col1.button("⬅️", help=f"Go to {prev_day['date']}", key="prev_day_btn", use_container_width=True):
                st.session_state.selected_date = prev_day["date"]
                st.rerun()
        
        # Display header in center column with compact styling
        col2.markdown(f'<h2 class="location-header">📍 {day["location"]}</h2>', unsafe_allow_html=True)
        col2.markdown(f'<h3 class="date-subheader">{day["date"]}</h3>', unsafe_allow_html=True)
        
        # Next day button with improved styling
        if current_index < len(trip_data) - 1:
            next_day = trip_data[current_index + 1]
            if col3.button("➡️", help=f"Go to {next_day['date']}", key="next_day_btn", use_container_width=True):
                st.session_state.selected_date = next_day["date"]
                st.rerun()
        
        # Add compact progress indicator with label
        progress = (current_index + 1) / len(trip_data)
        prog_col1, prog_col2 = st.columns([6, 1])
        with prog_col1:
            st.progress(progress, "")  # Empty string to remove label
        with prog_col2:
            st.markdown(f"<div style='text-align: center; font-size: 0.7rem; margin-top: -5px;'>{current_index + 1}/{len(trip_data)}</div>", unsafe_allow_html=True)
        
        # Weather information - Moved before itinerary details
        coordinates = get_location_coordinates(day["location"])
        if coordinates:
            lat, lon = coordinates
            weather = get_current_weather(lat, lon)
            if weather:
                icon = get_weather_icon(weather['weather'][0]['icon'])
                
                # Create a 2-column layout for current weather and 3-day forecast
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(
                        f"""<div class="content-card weather-card">
                            <h4>🌤️ Current Weather</h4>
                            <div class="card-content">
                                <div style="display: flex; align-items: center;">
                                    <div style="font-size: 3rem; margin-right: 15px;">
                                        {icon}
                                    </div>
                                    <div>
                                        <div style="font-weight: bold; font-size: 1.1rem;">{weather['weather'][0]['description'].capitalize()}</div>
                                        <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{weather['main']['temp']}°C</div>
                                        <div style="font-size: 0.9rem;">Feels like: {weather['main']['feels_like']}°C</div>
                                    </div>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                                    <div style="text-align: center; flex: 1;">
                                        <div style="font-size: 0.9rem;">Humidity</div>
                                        <div style="font-weight: bold;">{weather['main']['humidity']}%</div>
                                    </div>
                                    <div style="text-align: center; flex: 1;">
                                        <div style="font-size: 0.9rem;">Wind</div>
                                        <div style="font-weight: bold;">{weather['wind']['speed']} km/h</div>
                                    </div>
                                    <div style="text-align: center; flex: 1;">
                                        <div style="font-size: 0.9rem;">Pressure</div>
                                        <div style="font-weight: bold;">{weather['main']['pressure']} hPa</div>
                                    </div>
                                </div>
                            </div>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                
                # Get forecast data
                forecast = get_forecast(lat, lon)
                
                # Always display 3-day forecast in the second column
                with col2:
                    if forecast and 'daily' in forecast and len(forecast['daily']) > 0:
                        st.markdown(
                            f"""<div class='content-card weather-card' style='height: 100%;'>
                                <h4>📅 3-Day Forecast</h4>
                                <div class='card-content'>
                            """, unsafe_allow_html=True
                        )
                        cols = st.columns(3)
                        for i in range(3):
                            with cols[i]:
                                try:
                                    day_forecast = forecast['daily'][i]
                                    date = datetime.fromtimestamp(day_forecast['dt']).strftime("%a %d")
                                    icon_code = day_forecast['weather'][0]['icon'] if 'weather' in day_forecast and day_forecast['weather'] else '01d'
                                    icon = get_weather_icon(icon_code)
                                    if 'temp' in day_forecast and isinstance(day_forecast['temp'], dict):
                                        max_temp = day_forecast['temp'].get('max', 0)
                                        min_temp = day_forecast['temp'].get('min', 0)
                                    else:
                                        max_temp = day_forecast.get('temp_max', 0) if 'temp_max' in day_forecast else 0
                                        min_temp = day_forecast.get('temp_min', 0) if 'temp_min' in day_forecast else 0
                                    pop = day_forecast.get('pop', 0)
                                    pop_formatted = f"{pop * 100:.0f}%" if isinstance(pop, (int, float)) else "N/A"
                                    weather_desc = day_forecast['weather'][0]['description'].capitalize() if 'weather' in day_forecast and day_forecast['weather'] else 'Unknown'
                                    st.markdown(f"""
                                        <div class='forecast-day-compact' style='background: rgba(255, 255, 255, 0.95); border-radius: 8px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                            <div style='display: flex; flex-direction: column; align-items: center;'>
                                                <div style='font-weight: bold; font-size: 0.9rem; color: #333;'>{date}</div>
                                                <div style='font-size: 1.8rem;'>{icon}</div>
                                                <div style='font-size: 0.8rem; color: #555;'>{weather_desc}</div>
                                                <div style='margin: 2px 0;'>
                                                    <span style='color: #d63031; font-weight: bold;'>{max_temp:.1f}°C</span> /
                                                    <span style='color: #0984e3;'>{min_temp:.1f}°C</span>
                                                </div>
                                                <div style='font-size: 0.8rem; color: #555;'>🌧️ {pop_formatted}</div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown(f"""
                                        <div class='forecast-day-compact' style='background: rgba(255, 255, 255, 0.95); border-radius: 8px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                            <div style='display: flex; flex-direction: column; align-items: center;'>
                                                <div style='font-weight: bold; font-size: 0.9rem; color: #333;'>Day {i+1}</div>
                                                <div style='font-size: 1.8rem;'>🌡️</div>
                                                <div style='font-size: 0.8rem; color: #555;'>Unknown</div>
                                                <div style='margin: 2px 0; color: #555;'>--°C / --°C</div>
                                                <div style='font-size: 0.8rem; color: #555;'>--</div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    print(f"Error in 3-day forecast for day {i}: {e}")
                        st.markdown("""
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"""<div class="content-card weather-card">
                                <h4>📅 Forecast Unavailable</h4>
                                <div class="card-content" style="text-align: center;">
                                    <p>Weather forecast data is not available for this location.</p>
                                </div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                
                # Hourly forecast has been removed to fix HTML rendering issues
                
                # 5-day forecast has been removed to fix HTML rendering issues
            else:
                st.info("⚠️ Weather information is not available for this location.")
                st.markdown("""
                The application uses the Met Norway API to fetch weather data. 
                Weather information may not be available for all locations or may be temporarily unavailable.
                """)
        
        # Display details with improved formatting and more compact design
        st.markdown(
            f"""<div class="content-card">
                <h4>📅 Itinerary Details</h4>
                <div class="card-content">{day['details']}</div>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Personal trip features
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Mark as visited checkbox
            visited_key = f"visited_{day['date']}"
            is_visited = st.checkbox(
                "✅ Mark as Visited", 
                value=st.session_state.visited_status.get(day["date"], False),
                key=visited_key
            )
            st.session_state.visited_status[day["date"]] = is_visited
        
        with col2:
            # Add to favorites button
            is_favorite = day["location"] in st.session_state.favorite_places
            if st.button(
                f"{'💖 Remove from Favorites' if is_favorite else '🤍 Add to Favorites'}", 
                key=f"fav_{day['date']}"
            ):
                if is_favorite:
                    st.session_state.favorite_places.discard(day["location"])
                    st.success(f"Removed {day['location']} from favorites!")
                else:
                    st.session_state.favorite_places.add(day["location"])
                    st.success(f"Added {day['location']} to favorites!")
                st.rerun()
        
        with col3:
            # Weather preference indicator
            user_weather_pref = st.session_state.preferences.get('weather_preference', 'any')
            if user_weather_pref != 'any':
                weather_match = "☀️ Great weather match!" if user_weather_pref in day.get('weather_keywords', []) else "🌤️ Check weather"
                st.info(weather_match)
        
        # Personal notes section
        st.markdown(
            f"""<div class="content-card">
                <h4>📝 Personal Notes & Memories</h4>
                <div class="card-content">
            """, 
            unsafe_allow_html=True
        )
        
        # Text area for personal notes
        note_key = f"note_{day['date']}"
        current_note = st.session_state.personal_notes.get(day["date"], "")
        
        personal_note = st.text_area(
            "Add your thoughts, memories, or planning notes:",
            value=current_note,
            height=100,
            key=note_key,
            placeholder="Write about your experiences, things to remember, or planning notes..."
        )
        
        if personal_note != current_note:
            st.session_state.personal_notes[day["date"]] = personal_note
        
        # Quick note buttons
        quick_notes = ["Amazing views!", "Great food", "Weather was perfect", "Challenging hike", "Very crowded", "Hidden gem"]
        
        st.markdown("**Quick notes:**")
        cols = st.columns(6)  # Use 6 columns for a single row
        for i, quick_note in enumerate(quick_notes):
            with cols[i]:
                if st.button(quick_note, key=f"quick_{day['date']}_{i}"):
                    current = st.session_state.personal_notes.get(day["date"], "")
                    if quick_note not in current:
                        new_note = f"{current}\n• {quick_note}".strip()
                        st.session_state.personal_notes[day["date"]] = new_note
                        st.rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Add a prominent personalization summary if user has customized preferences
        user_activities = st.session_state.preferences.get('preferred_activities', [])
        fitness_level = st.session_state.preferences.get('fitness_level', 'moderate')
        budget_level = st.session_state.preferences.get('budget_level', 'medium')
        dietary_restrictions = st.session_state.preferences.get('dietary_restrictions', [])
        
        has_customizations = (
            user_activities or 
            fitness_level != 'moderate' or 
            budget_level != 'medium' or 
            (dietary_restrictions and "None" not in dietary_restrictions)
        )
        
        if has_customizations:
            st.markdown(
                f"""<div class="content-card" style="background: linear-gradient(135deg, #E8F5E8 0%, #E3F2FD 100%); border: 3px solid #4CAF50; box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);">
                    <h4>🎯 Your Personal Trip Profile - Active</h4>
                    <div class="card-content">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px;">
                            <div style="background: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 6px;">
                                <strong>🎪 Activities You Love:</strong><br>
                                <span style="color: #2E7D32; font-weight: 600;">{', '.join(user_activities) if user_activities else 'Not specified'}</span>
                            </div>
                            <div style="background: rgba(33, 150, 243, 0.1); padding: 10px; border-radius: 6px;">
                                <strong>💪 Fitness Level:</strong><br>
                                <span style="color: #1976D2; font-weight: 600;">{fitness_level.title()}</span>
                            </div>
                            <div style="background: rgba(255, 193, 7, 0.1); padding: 10px; border-radius: 6px;">
                                <strong>💰 Budget Preference:</strong><br>
                                <span style="color: #F57C00; font-weight: 600;">{budget_level.title()}</span>
                            </div>
                            <div style="background: rgba(156, 39, 176, 0.1); padding: 10px; border-radius: 6px;">
                                <strong>🍽️ Dietary Needs:</strong><br>
                                <span style="color: #7B1FA2; font-weight: 600;">{', '.join(dietary_restrictions) if dietary_restrictions and 'None' not in dietary_restrictions else 'None specified'}</span>
                            </div>
                        </div>
                    </div>
                </div>""", 
                unsafe_allow_html=True
            )
            
            # Make the green box clickable to toggle examples
            if st.button("👀 See Examples of Personalization in Action", 
                       key="toggle_examples", 
                       use_container_width=True,
                       help="Click to see personalization examples"):
                # Toggle the examples visibility
                if 'show_examples' not in st.session_state:
                    st.session_state.show_examples = False
                st.session_state.show_examples = not st.session_state.show_examples
        else:
            st.markdown(
                f"""<div class="content-card" style="background: linear-gradient(135deg, #FFF3E0 0%, #FFECB3 100%); border: 2px solid #FF9800;">
                    <h4>⚙️ Trip Customization Available</h4>
                    <div class="card-content">
                        <div style="text-align: center; padding: 20px;">
                            <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
                            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 10px; color: #E65100;">
                                Get Personalized Recommendations!
                            </div>
                            <div style="margin-bottom: 15px; color: #BF360C;">
                                Set your preferences in the sidebar to see custom suggestions for:
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0;">
                                <div style="background: rgba(255, 152, 0, 0.1); padding: 8px; border-radius: 6px;">🥾 Activity matching</div>
                                <div style="background: rgba(255, 152, 0, 0.1); padding: 8px; border-radius: 6px;">🍽️ Dining suggestions</div>
                                <div style="background: rgba(255, 152, 0, 0.1); padding: 8px; border-radius: 6px;">💪 Fitness recommendations</div>
                                <div style="background: rgba(255, 152, 0, 0.1); padding: 8px; border-radius: 6px;">💰 Budget guidance</div>
                            </div>
                            <div style="margin-top: 15px; padding: 8px; background: rgba(255, 152, 0, 0.2); border-radius: 6px; font-weight: 600;">
                                👈 Open "Customize Your Trip" in the sidebar to get started!
                            </div>
                        </div>
                    </div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        # Show example of what personalization looks like - only when toggled
        if st.session_state.get('show_examples', False):
            st.markdown("### 📚 Personalization Examples")
            with st.container():
                st.markdown("""
                **Example 1: Moderate Fitness + Hiking Interest**
                - Day 14 (Kjerag): "⚠️ Very challenging hikes - consider shorter alternatives or guided tours"
                - Day 4 (Beaches): "🚶‍♂️ Consider easier hiking trails for coastal walks"
                
                **Example 2: Very Active + Wildlife Viewing Interest**  
                - Day 6 (Nusfjord): "🦅 Exceptional wildlife viewing opportunities - bring binoculars!"
                - Day 14 (Kjerag): "� These challenging hikes are perfect for your fitness level!"
                
                **Example 3: Budget Level + Food Tours Interest**
                - Budget: "💰 Look for free activities and local markets for affordable meals"
                - Luxury: "🍾 Don't miss Michelin-recommended restaurants!"
                
                **Example 4: Dietary Restrictions**
                - Vegetarian: "🥗 Norway has great vegetarian options - try local root vegetables"
                - Gluten-free: "🌾 Most restaurants can accommodate - ask for 'glutenfri'"
                
                **🎯 Try it yourself:**
                1. Set preferences in the sidebar
                2. Navigate to Day 14 (Kjerag Hike) or Day 6 (Nusfjord)  
                3. Look for colored recommendation boxes below activities!
                """)
                
                st.success("💡 **Pro tip:** Different combinations create unique recommendations for each day!")
                
                # Additional dining examples
                st.markdown("---")
                st.markdown("**🍽️ SPECIFIC DINING EXAMPLES:**")
                st.markdown("""
                **Budget + Vegetarian in Bergen:** "🥗 Bergen has excellent vegetarian cafes in the old town"
                
                **Luxury + Seafood Allergy in Lofoten:** "🥔 Focus on local potatoes and root vegetables - Lofoten specialties!"
                
                **Premium + Gluten-free in Stavanger:** "👨‍🍳 High-end restaurants excel at gluten-free fine dining"
                """)
        
        # Display activities and dining options side by side for better space efficiency
        has_activities = 'activities' in day and day['activities']
        has_dining = 'dining_options' in day and day['dining_options']
        
        if has_activities or has_dining:
            # Create two-column layout for activities and dining
            if has_activities and has_dining:
                # Both sections available - use equal columns
                activity_col, dining_col = st.columns([1, 1])
            elif has_activities:
                # Only activities - use full width
                activity_col = st.container()
                dining_col = None
            else:
                # Only dining - use full width
                activity_col = None
                dining_col = st.container()
            
            # Display activities
            if has_activities:
                with activity_col if activity_col else st.container():
                    # Get user preferences to customize activity display
                    user_activities = st.session_state.preferences.get('preferred_activities', [])
                    fitness_level = st.session_state.preferences.get('fitness_level', 'moderate')
                    budget_level = st.session_state.preferences.get('budget_level', 'medium')
                    
                    # Get place information with ratings for activities
                    location_context = day["location"].split('→')[0].strip() if '→' in day["location"] else day["location"]
                    
                    # Customize activities based on preferences
                    customized_activities = []
                    base_activities = day['activities'].copy()
                    
                    # Add additional personalized activities based on preferences
                    location_lower = day["location"].lower()
                    
                    # Add preference-based activities
                    if "Photography" in user_activities:
                        if "bergen" in location_lower and "Photography walk in old town" not in base_activities:
                            customized_activities.append("📸 Photography walk in old town")
                        elif "lofoten" in location_lower and "Golden hour photography" not in base_activities:
                            customized_activities.append("📸 Golden hour photography")
                        elif "fjord" in location_lower and "Scenic photography spots" not in base_activities:
                            customized_activities.append("📸 Scenic photography spots")
                    
                    if "Museums" in user_activities:
                        if "bergen" in location_lower and "Bergen Art Museums" not in base_activities:
                            customized_activities.append("🏛️ Bergen Art Museums")
                        elif "stavanger" in location_lower and "Norwegian Petroleum Museum" not in base_activities:
                            customized_activities.append("🏛️ Norwegian Petroleum Museum")
                    
                    if "Food Tours" in user_activities:
                        if "bergen" in location_lower and "Fish market food tour" not in base_activities:
                            customized_activities.append("🍽️ Fish market food tour")
                        elif "lofoten" in location_lower and "Local seafood tasting" not in base_activities:
                            customized_activities.append("🍽️ Local seafood tasting")
                    
                    if "Wildlife Viewing" in user_activities:
                        if "lofoten" in location_lower and "Sea eagle watching" not in base_activities:
                            customized_activities.append("🦅 Sea eagle watching")
                        elif "fjord" in location_lower and "Seal spotting" not in base_activities:
                            customized_activities.append("🦭 Seal spotting")
                    
                    # Fitness-based activity modifications
                    if fitness_level == "light":
                        if "henningsvær" in location_lower and "Art gallery walking tour" not in base_activities:
                            customized_activities.append("🎨 Art gallery walking tour")
                        if "bergen" in location_lower and "Funicular ride to Mount Fløyen" not in base_activities:
                            customized_activities.append("🚡 Funicular ride to Mount Fløyen")
                    elif fitness_level in ["active", "very_active"]:
                        if any(keyword in location_lower for keyword in ["kjerag", "pulpit"]):
                            customized_activities.append("🧗‍♂️ Extended hiking routes")
                        if "lofoten" in location_lower and "Challenging coastal hikes" not in base_activities:
                            customized_activities.append("🥾 Challenging coastal hikes")
                    
                    # Budget-based activity suggestions
                    if budget_level == "budget":
                        customized_activities.append("💰 Free viewpoint walks")
                        if "bergen" in location_lower:
                            customized_activities.append("🚶‍♂️ Self-guided walking tour")
                    elif budget_level == "luxury":
                        if "bergen" in location_lower:
                            customized_activities.append("✨ Private helicopter tour")
                        elif "fjord" in location_lower:
                            customized_activities.append("🛥️ Private fjord cruise")
                    
                    # Combine base activities with customized ones
                    all_activities = base_activities + customized_activities
                    
                    # Sort activities by user preference relevance
                    prioritized_activities = []
                    other_activities = []
                    
                    for activity in all_activities:
                        activity_lower = activity.lower()
                        is_preferred = False
                        
                        if "Hiking" in user_activities and any(keyword in activity_lower for keyword in ["hik", "walk", "trail", "climb"]):
                            is_preferred = True
                        elif "Photography" in user_activities and any(keyword in activity_lower for keyword in ["photo", "view", "scenic"]):
                            is_preferred = True
                        elif "Museums" in user_activities and any(keyword in activity_lower for keyword in ["museum", "gallery", "art"]):
                            is_preferred = True
                        elif "Food Tours" in user_activities and any(keyword in activity_lower for keyword in ["food", "market", "tasting"]):
                            is_preferred = True
                        elif "Wildlife Viewing" in user_activities and any(keyword in activity_lower for keyword in ["wildlife", "eagle", "seal", "whale"]):
                            is_preferred = True
                        elif "Scenic Drives" in user_activities and any(keyword in activity_lower for keyword in ["drive", "scenic", "route"]):
                            is_preferred = True
                        elif "Beaches" in user_activities and any(keyword in activity_lower for keyword in ["beach", "swim", "coast"]):
                            is_preferred = True
                        elif "Adventure Sports" in user_activities and any(keyword in activity_lower for keyword in ["climb", "extreme", "adventure"]):
                            is_preferred = True
                        
                        if is_preferred:
                            prioritized_activities.append(activity)
                        else:
                            other_activities.append(activity)
                    
                    # Final activity list with preferred activities first
                    final_activities = prioritized_activities + other_activities
                    
                    activities_with_details = []
                    
                    for i, activity in enumerate(final_activities):
                        # Get Google Maps URL and rating
                        maps_url, rating = get_place_details(activity, location_context)
                        rating_stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
                        
                        # Add activity-specific icons
                        activity_icon = "🚶‍♂️"  # default
                        activity_lower = activity.lower()
                        if "hik" in activity_lower or "climb" in activity_lower:
                            activity_icon = "🥾"
                        elif "swim" in activity_lower or "beach" in activity_lower:
                            activity_icon = "🏊‍♂️"
                        elif "photo" in activity_lower or "view" in activity_lower:
                            activity_icon = "📸"
                        elif "drive" in activity_lower or "scenic" in activity_lower:
                            activity_icon = "🚗"
                        elif "shop" in activity_lower:
                            activity_icon = "🛍️"
                        elif "museum" in activity_lower or "gallery" in activity_lower:
                            activity_icon = "🏛️"
                        elif "cruise" in activity_lower or "boat" in activity_lower:
                            activity_icon = "⛵"
                        elif "flight" in activity_lower or "airport" in activity_lower:
                            activity_icon = "✈️"
                        elif "eagle" in activity_lower or "wildlife" in activity_lower:
                            activity_icon = "🦅"
                        elif "food" in activity_lower or "tasting" in activity_lower:
                            activity_icon = "🍽️"
                        elif "art" in activity_lower:
                            activity_icon = "🎨"
                        elif "helicopter" in activity_lower:
                            activity_icon = "🚁"
                        elif "funicular" in activity_lower:
                            activity_icon = "🚡"
                        
                        # Determine if this activity matches user preferences (for highlighting)
                        is_match = i < len(prioritized_activities)
                        match_style = "background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%); border: 2px solid #4CAF50;" if is_match else ""
                        match_label = " ⭐ MATCHES YOUR INTERESTS" if is_match and activity not in base_activities else ""
                        
                        # Create HTML with rating and link
                        activity_html = f"""<li>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 6px; border-radius: 6px; {match_style}" class="clickable-item activity-item">
                                <div style="flex: 1; margin-right: 8px;">
                                    <span class="activity-icon">{activity_icon}</span>{activity}{match_label}
                                </div>
                                <div style="display: flex; align-items: center; flex-shrink: 0;">
                                    <span style="color: #FFD700; margin-right: 3px; font-size: 0.8rem;">{rating_stars}</span>
                                    <span style="color: #666; font-size: 0.8rem; margin-right: 8px;">{rating}/5</span>
                                    <a href="{maps_url}" target="_blank" style="display: inline-flex; align-items: center; justify-content: center; background-color: #4285F4; color: white; width: 24px; height: 24px; border-radius: 50%; text-decoration: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s; font-size: 0.7rem;">🗺️</a>
                                </div>
                            </div>
                        </li>"""
                        activities_with_details.append(activity_html)
                    
                    # Join all activities HTML
                    activities_html = ''.join(activities_with_details)
                    
                    # Add personalized recommendations
                    recommendations = []
                    location_lower = day["location"].lower()
                    
                    # Smart recommendations based on preferences and location
                    if "Hiking" in user_activities and any(keyword in location_lower for keyword in ["mountain", "hike", "cliff", "rock"]):
                        if fitness_level in ["active", "very_active"]:
                            recommendations.append("🥾 Perfect for challenging hikes - matches your fitness level!")
                        elif fitness_level == "moderate":
                            recommendations.append("🚶‍♂️ Consider easier hiking trails based on your fitness preference")
                        else:  # light
                            recommendations.append("⚠️ This location has challenging hikes - consider scenic drives instead")
                    
                    if "Photography" in user_activities and any(keyword in location_lower for keyword in ["view", "scenic", "fjord", "waterfall"]):
                        recommendations.append("📸 Excellent photography opportunities - bring extra batteries!")
                        if "sunrise" in ' '.join(day.get('activities', [])).lower():
                            recommendations.append("🌅 Early morning golden hour shots recommended!")
                    
                    if "Beaches" in user_activities and "beach" in location_lower:
                        recommendations.append("🏖️ Beach day! Perfect for your interests")
                        if fitness_level in ["active", "very_active"]:
                            recommendations.append("🏃‍♂️ Great for beach running and water sports!")
                    
                    if "Wildlife Viewing" in user_activities:
                        if any(keyword in location_lower for keyword in ["safari", "eagle", "wildlife"]):
                            recommendations.append("🦅 Exceptional wildlife viewing opportunities - bring binoculars!")
                        elif "lofoten" in location_lower:
                            recommendations.append("🐋 Watch for whales and sea birds in these waters!")
                        elif "fjord" in location_lower:
                            recommendations.append("🦭 Seals and porpoises often spotted in fjords!")
                    
                    if "Scenic Drives" in user_activities and any(keyword in location_lower for keyword in ["drive", "scenic", "route"]):
                        recommendations.append("🚗 Perfect scenic driving day - plan for photo stops!")
                        if fitness_level == "light":
                            recommendations.append("🚗 Great option for enjoying Norway's beauty at your own pace")
                    
                    if "Museums" in user_activities and any(keyword in location_lower for keyword in ["museum", "gallery", "cultural"]):
                        recommendations.append("🏛️ Cultural exploration matches your interests perfectly!")
                    
                    if "Adventure Sports" in user_activities:
                        if any(keyword in location_lower for keyword in ["canyon", "climb", "extreme"]):
                            if fitness_level in ["active", "very_active"]:
                                recommendations.append("🧗‍♂️ Adventure sports available - perfect for your fitness level!")
                            else:
                                recommendations.append("⚠️ Adventure sports here require high fitness - consider alternatives")
                    
                    if "Food Tours" in user_activities and any(keyword in location_lower for keyword in ["restaurant", "market", "food"]):
                        dietary = st.session_state.preferences.get('dietary_restrictions', [])
                        if dietary and "None" not in dietary:
                            recommendations.append(f"🍽️ Great food scene! Remember your dietary needs: {', '.join(dietary)}")
                        else:
                            recommendations.append("🍽️ Excellent culinary experiences await!")
                    
                    # Fitness-specific recommendations for specific challenging days
                    if day["date"] in ["2025-08-14 (Thursday)", "2025-08-15 (Friday)"]:  # Kjerag and Pulpit Rock
                        if fitness_level == "very_active":
                            recommendations.append("💪 These challenging hikes are perfect for your fitness level!")
                        elif fitness_level == "active":
                            recommendations.append("🥾 Challenging but doable hikes - take your time!")
                        elif fitness_level == "moderate":
                            recommendations.append("⚠️ Very challenging hikes - consider shorter alternatives or guided tours")
                        else:  # light
                            recommendations.append("🚗 Consider scenic drives to viewpoints instead of hiking")
                    
                    # Location-specific activity matching
                    if "henningsvær" in location_lower and "Museums" in user_activities:
                        recommendations.append("🎨 Art galleries in Henningsvær are world-class!")
                    
                    if "bergen" in location_lower and "Food Tours" in user_activities:
                        recommendations.append("🐟 Bergen's fish market is a food lover's paradise!")
                    
                    # Weather-based recommendations for active users
                    if fitness_level in ["active", "very_active"] and any(keyword in location_lower for keyword in ["outdoor", "hike", "beach"]):
                        recommendations.append("☀️ Check weather for optimal outdoor activity timing")
                    
                    # Budget-based recommendations
                    budget = st.session_state.preferences.get('budget_level', 'medium')
                    if budget == "budget":
                        recommendations.append("💰 Look for free activities and local markets for affordable meals")
                        if "Photography" in user_activities:
                            recommendations.append("📸 Free photography spots - no entrance fees needed!")
                    elif budget == "luxury":
                        recommendations.append("✨ Consider premium experiences and fine dining options")
                        if "Food Tours" in user_activities:
                            recommendations.append("🍾 Don't miss Michelin-recommended restaurants!")
                    elif budget == "premium":
                        if "Adventure Sports" in user_activities:
                            recommendations.append("🎿 Consider guided adventure tours for safety and expertise")
                    
                    # Display personalized recommendations header
                    if recommendations:
                        st.markdown(
                            f"""<div style="background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%); 
                                border: 2px solid #4CAF50; padding: 15px; margin: 15px 0; border-radius: 8px;
                                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);">
                                <h5 style="margin: 0 0 10px 0; color: #2E7D32;">🎯 Personalized Activity Tips for You</h5>
                                <div style="font-size: 0.95rem;">
                                    {'<br>'.join(['• ' + rec for rec in recommendations])}
                                </div>
                                <div style="margin-top: 10px; font-size: 0.85rem; color: #1B5E20; font-style: italic;">
                                    ✨ Activities marked with ⭐ match your interests!
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    # Show activity personalization summary
                    added_activities = [act for act in final_activities if act not in base_activities]
                    if added_activities:
                        st.markdown(
                            f"""<div style="background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%); 
                                border: 1px solid #2196F3; padding: 10px; margin: 10px 0; border-radius: 6px;">
                                <div style="font-size: 0.9rem; color: #1976D2;">
                                    <strong>✨ {len(added_activities)} activities added based on your preferences:</strong><br>
                                    {', '.join(added_activities)}
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    st.markdown(
                        f"""<div class="content-card">
                            <h4>🚶‍♂️ Activities</h4>
                            <div class="card-content interactive-card">
                                <ul style="list-style-type: none; padding-left: 0;">{activities_html}</ul>
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
            
            # Display dining options
            if has_dining:
                with dining_col if dining_col else st.container():
                    # Get user preferences to customize dining display
                    dietary_restrictions = st.session_state.preferences.get('dietary_restrictions', [])
                    budget_level = st.session_state.preferences.get('budget_level', 'medium')
                    food_interests = "Food Tours" in st.session_state.preferences.get('preferred_activities', [])
                    
                    # Get place information with ratings for dining options
                    location_context = day["location"].split('→')[0].strip() if '→' in day["location"] else day["location"]
                    location_lower = day["location"].lower()
                    
                    # Start with base dining options
                    customized_dining = day['dining_options'].copy()
                    
                    # Add personalized dining options based on preferences
                    if budget_level == "budget":
                        if "grocery shopping" not in [d.lower() for d in customized_dining]:
                            customized_dining.append("🛒 Local grocery stores")
                        if "market" not in location_lower and "market food stalls" not in [d.lower() for d in customized_dining]:
                            customized_dining.append("🏪 Market food stalls")
                    elif budget_level == "luxury":
                        if "bergen" in location_lower and "michelin" not in " ".join(customized_dining).lower():
                            customized_dining.append("⭐ Michelin-starred dining")
                        elif "stavanger" in location_lower and "fine dining" not in " ".join(customized_dining).lower():
                            customized_dining.append("🍾 Fine dining restaurants")
                        if "private chef" not in " ".join(customized_dining).lower():
                            customized_dining.append("👨‍🍳 Private chef experience")
                    elif budget_level == "premium":
                        if "tasting menu" not in " ".join(customized_dining).lower():
                            customized_dining.append("🍷 Tasting menu restaurants")
                    
                    # Add dietary-specific options
                    if "Vegetarian" in dietary_restrictions:
                        if "vegetarian" not in " ".join(customized_dining).lower():
                            customized_dining.append("🥗 Vegetarian-friendly restaurants")
                        if "bergen" in location_lower:
                            customized_dining.append("🌿 Bergen vegetarian cafes")
                    
                    if "Vegan" in dietary_restrictions:
                        if "vegan" not in " ".join(customized_dining).lower():
                            customized_dining.append("🌱 Vegan restaurants")
                        if budget_level == "budget":
                            customized_dining.append("🥬 Health food stores")
                    
                    if "Gluten-free" in dietary_restrictions:
                        if "gluten" not in " ".join(customized_dining).lower():
                            customized_dining.append("🌾 Gluten-free bakeries")
                        if budget_level != "budget":
                            customized_dining.append("🍞 Certified gluten-free restaurants")
                    
                    if "Seafood allergy" in dietary_restrictions:
                        # Emphasize non-seafood options
                        if "meat restaurant" not in " ".join(customized_dining).lower():
                            customized_dining.append("🥩 Traditional meat restaurants")
                        if "pizza" not in " ".join(customized_dining).lower():
                            customized_dining.append("🍕 Pizza places (safe from seafood)")
                    
                    # Add food tour specific options
                    if food_interests:
                        if "bergen" in location_lower and "food tour" not in " ".join(customized_dining).lower():
                            customized_dining.append("🍽️ Guided food tours")
                        if "market" in location_lower and "food sampling" not in " ".join(customized_dining).lower():
                            customized_dining.append("🥘 Market food sampling")
                        if "cooking class" not in " ".join(customized_dining).lower():
                            customized_dining.append("👩‍🍳 Norwegian cooking classes")
                    
                    # Location-specific additions
                    if "lofoten" in location_lower:
                        if any(restriction in dietary_restrictions for restriction in ["Seafood allergy"]):
                            customized_dining.append("🥔 Local potato dishes (safe option)")
                        elif budget_level == "luxury":
                            customized_dining.append("🦞 Premium seafood experiences")
                    
                    if "bergen" in location_lower:
                        if budget_level == "budget" and "street food" not in " ".join(customized_dining).lower():
                            customized_dining.append("🌭 Bergen street food")
                        elif budget_level == "luxury":
                            customized_dining.append("🍸 Rooftop dining with fjord views")
                    
                    # Sort dining options by relevance to user preferences
                    prioritized_dining = []
                    other_dining = []
                    
                    base_count = len(day['dining_options'])
                    
                    for i, dining in enumerate(customized_dining):
                        dining_lower = dining.lower()
                        is_personalized = i >= base_count
                        is_dietary_match = False
                        is_budget_match = False
                        
                        # Check dietary matches
                        if "None" not in dietary_restrictions and dietary_restrictions:
                            if ("Vegetarian" in dietary_restrictions and any(keyword in dining_lower for keyword in ["vegetarian", "vegan", "plant"])) or \
                               ("Vegan" in dietary_restrictions and "vegan" in dining_lower) or \
                               ("Gluten-free" in dietary_restrictions and "gluten" in dining_lower) or \
                               ("Seafood allergy" in dietary_restrictions and any(keyword in dining_lower for keyword in ["meat", "pizza", "potato"])):
                                is_dietary_match = True
                        
                        # Check budget matches
                        if (budget_level == "budget" and any(keyword in dining_lower for keyword in ["grocery", "market", "street"])) or \
                           (budget_level == "luxury" and any(keyword in dining_lower for keyword in ["michelin", "fine", "private", "premium"])) or \
                           (budget_level == "premium" and "tasting" in dining_lower):
                            is_budget_match = True
                        
                        if is_personalized or is_dietary_match or is_budget_match:
                            prioritized_dining.append((dining, is_personalized, is_dietary_match, is_budget_match))
                        else:
                            other_dining.append((dining, False, False, False))
                    
                    # Combine prioritized and other dining options
                    final_dining = prioritized_dining + other_dining
                    
                    dining_with_details = []
                    
                    for dining_info in final_dining:
                        dining, is_personalized, is_dietary_match, is_budget_match = dining_info
                        
                        # Get Google Maps URL and rating
                        maps_url, rating = get_place_details(dining, location_context)
                        rating_stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
                        
                        # Add dining-specific icons
                        dining_icon = "🍽️"  # default
                        dining_lower = dining.lower()
                        if "seafood" in dining_lower or "fish" in dining_lower:
                            dining_icon = "🐟"
                        elif "restaurant" in dining_lower:
                            dining_icon = "🍴"
                        elif "cafe" in dining_lower or "coffee" in dining_lower:
                            dining_icon = "☕"
                        elif "market" in dining_lower or "grocery" in dining_lower:
                            dining_icon = "🏪"
                        elif "breakfast" in dining_lower:
                            dining_icon = "🥐"
                        elif "lunch" in dining_lower or "picnic" in dining_lower:
                            dining_icon = "🥪"
                        elif "dinner" in dining_lower:
                            dining_icon = "🍽️"
                        elif "snack" in dining_lower:
                            dining_icon = "🥨"
                        elif "brewery" in dining_lower or "beer" in dining_lower:
                            dining_icon = "🍺"
                        elif "flight" in dining_lower or "airport" in dining_lower:
                            dining_icon = "✈️"
                        elif "vegetarian" in dining_lower or "vegan" in dining_lower:
                            dining_icon = "🥗"
                        elif "gluten" in dining_lower:
                            dining_icon = "🌾"
                        elif "meat" in dining_lower:
                            dining_icon = "🥩"
                        elif "pizza" in dining_lower:
                            dining_icon = "🍕"
                        elif "michelin" in dining_lower or "fine" in dining_lower:
                            dining_icon = "⭐"
                        elif "cooking" in dining_lower:
                            dining_icon = "👩‍🍳"
                        elif "food tour" in dining_lower:
                            dining_icon = "🍽️"
                        
                        # Determine highlighting based on personalization
                        highlight_style = ""
                        match_labels = []
                        
                        if is_personalized:
                            highlight_style = "background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%); border: 2px solid #4CAF50;"
                            match_labels.append("✨ ADDED FOR YOU")
                        elif is_dietary_match:
                            highlight_style = "background: linear-gradient(135deg, #FFF3E0 0%, #E1F5FE 100%); border: 2px solid #FF9800;"
                            match_labels.append("🍽️ DIETARY MATCH")
                        elif is_budget_match:
                            highlight_style = "background: linear-gradient(135deg, #F3E5F5 0%, #E8F5E8 100%); border: 2px solid #9C27B0;"
                            match_labels.append("💰 BUDGET MATCH")
                        
                        match_label_text = " " + " ".join(match_labels) if match_labels else ""
                        
                        # Create HTML with rating and link
                        dining_html = f"""<li>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 6px; border-radius: 6px; {highlight_style}" class="clickable-item dining-item">
                                <div style="flex: 1; margin-right: 8px;">
                                    <span class="dining-icon">{dining_icon}</span>{dining}{match_label_text}
                                </div>
                                <div style="display: flex; align-items: center; flex-shrink: 0;">
                                    <span style="color: #FFD700; margin-right: 3px; font-size: 0.8rem;">{rating_stars}</span>
                                    <span style="color: #666; font-size: 0.8rem; margin-right: 8px;">{rating}/5</span>
                                    <a href="{maps_url}" target="_blank" style="display: inline-flex; align-items: center; justify-content: center; background-color: #4285F4; color: white; width: 24px; height: 24px; border-radius: 50%; text-decoration: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s; font-size: 0.7rem;">🗺️</a>
                                </div>
                            </div>
                        </li>"""
                        dining_with_details.append(dining_html)
                    
                    # Join all dining HTML
                    dining_html = ''.join(dining_with_details)
                    
                    # Add personalized dining recommendations
                    dining_recommendations = []
                    
                    # Dietary-specific recommendations
                    if dietary_restrictions and "None" not in dietary_restrictions:
                        dining_recommendations.append(f"⚠️ Remember your dietary needs: {', '.join(dietary_restrictions)}")
                        
                        # Specific suggestions based on restrictions
                        if "Vegetarian" in dietary_restrictions:
                            dining_recommendations.append("🥗 Norway has great vegetarian options - try local root vegetables and fish alternatives")
                            if "bergen" in location_lower:
                                dining_recommendations.append("🥕 Bergen fish market has fresh vegetable stalls too!")
                        if "Vegan" in dietary_restrictions:
                            dining_recommendations.append("🌱 Look for 'vegansk' on menus - Norwegian cities have growing vegan scenes")
                            if budget_level == "budget":
                                dining_recommendations.append("🛒 Grocery stores like Rema 1000 have good vegan options")
                        if "Gluten-free" in dietary_restrictions:
                            dining_recommendations.append("🌾 Most restaurants can accommodate gluten-free requests - ask for 'glutenfri'")
                            if budget_level != "budget":
                                dining_recommendations.append("🍞 Many upscale restaurants have dedicated gluten-free menus")
                        if "Seafood allergy" in dietary_restrictions:
                            dining_recommendations.append("🚫🐟 Be extra careful in coastal areas - inform restaurants about fish/shellfish allergies")
                            if "lofoten" in location_lower or "bergen" in location_lower:
                                dining_recommendations.append("⚠️ This is a major fishing area - be especially cautious")
                        if "Nut allergy" in dietary_restrictions:
                            dining_recommendations.append("🥜 Norwegian pastries often contain nuts - always ask 'inneholder dette nøtter?'")
                    
                    # Budget-specific dining advice with dietary considerations
                    if budget_level == "budget":
                        dining_recommendations.append("💰 Try local bakeries, food trucks, and grocery stores for budget-friendly meals")
                        dining_recommendations.append("🥪 Pack lunches for hiking days to save money")
                        if "Vegetarian" in dietary_restrictions:
                            dining_recommendations.append("🥖 Bakeries have great veggie sandwiches at good prices")
                        if "Gluten-free" in dietary_restrictions:
                            dining_recommendations.append("🛒 ICA and Coop supermarkets have affordable gluten-free sections")
                    elif budget_level == "luxury":
                        dining_recommendations.append("✨ Don't miss Michelin-recommended restaurants and local fine dining")
                        dining_recommendations.append("🍷 Try premium Norwegian wines and craft cocktails")
                        if "Vegetarian" in dietary_restrictions:
                            dining_recommendations.append("🌟 Norway's fine dining scene has incredible vegetarian tasting menus")
                        if "Gluten-free" in dietary_restrictions:
                            dining_recommendations.append("👨‍🍳 High-end restaurants excel at gluten-free fine dining")
                    elif budget_level == "premium":
                        dining_recommendations.append("🍽️ Mix of quality restaurants and local favorites recommended")
                        if dietary_restrictions and "None" not in dietary_restrictions:
                            dining_recommendations.append("📞 Call ahead to premium restaurants - they can accommodate most dietary needs")
                    
                    # Location-specific dining advice with personal considerations
                    location_lower = day["location"].lower()
                    if "lofoten" in location_lower:
                        if "Seafood allergy" not in dietary_restrictions:
                            dining_recommendations.append("🐟 Fresh Arctic seafood is exceptional here - try Arctic char or king crab")
                        else:
                            dining_recommendations.append("🥔 Focus on local potatoes and root vegetables - Lofoten specialties!")
                        if budget_level == "budget":
                            dining_recommendations.append("🎣 Some places let you buy directly from fishermen")
                    elif "bergen" in location_lower:
                        if "Seafood allergy" not in dietary_restrictions:
                            dining_recommendations.append("🦐 Fish market is a must-visit - try fresh shrimp and salmon")
                        if "Vegetarian" in dietary_restrictions:
                            dining_recommendations.append("🥗 Bergen has excellent vegetarian cafes in the old town")
                        if budget_level == "budget":
                            dining_recommendations.append("🐟 Fish market has affordable ready-to-eat options")
                    elif "stavanger" in location_lower:
                        if "Vegetarian" not in dietary_restrictions:
                            dining_recommendations.append("🧈 Try traditional Norwegian butter and local cheese specialties")
                        if budget_level == "luxury":
                            dining_recommendations.append("⭐ Stavanger has several Michelin-recommended restaurants")
                    elif "ålesund" in location_lower:
                        dining_recommendations.append("🦀 Known for king crab and Art Nouveau dining experiences")
                        if budget_level == "budget":
                            dining_recommendations.append("🥘 Local fish soup is affordable and filling")
                    elif "geiranger" in location_lower:
                        if budget_level == "budget":
                            dining_recommendations.append("🥪 Limited dining options - consider packing meals")
                        dining_recommendations.append("🏔️ Hotel restaurants often have the best food in remote areas")
                    
                    # Combine dietary + budget + location for specific recommendations
                    if budget_level == "budget" and "Vegetarian" in dietary_restrictions and "lofoten" in location_lower:
                        dining_recommendations.append("🌿 Stock up on vegetables in larger towns before heading to remote areas")
                    
                    # Show dining personalization summary
                    added_dining = [dining for dining, _, _, _ in final_dining if dining not in day['dining_options']]
                    if added_dining:
                        st.markdown(
                            f"""<div style="background: linear-gradient(135deg, #FFF3E0 0%, #F3E5F5 100%); 
                                border: 1px solid #FF9800; padding: 10px; margin: 10px 0; border-radius: 6px;">
                                <div style="font-size: 0.9rem; color: #E65100;">
                                    <strong>✨ {len(added_dining)} dining options added for your preferences:</strong><br>
                                    {', '.join(added_dining)}
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    if dining_recommendations:
                        st.markdown(
                            f"""<div style="background: linear-gradient(135deg, #FFF3E0 0%, #F3E5F5 100%); 
                                border: 2px solid #FF9800; padding: 15px; margin: 15px 0; border-radius: 8px;
                                box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);">
                                <h5 style="margin: 0 0 10px 0; color: #E65100;">🍴 Personalized Dining Tips for You</h5>
                                <div style="font-size: 0.95rem;">
                                    {'<br>'.join(['• ' + rec for rec in dining_recommendations])}
                                </div>
                                <div style="margin-top: 10px; font-size: 0.85rem; color: #BF360C; font-style: italic;">
                                    ✨ Highlighted options match your dietary needs and budget!
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    st.markdown(
                        f"""<div class="content-card">
                            <h4>🍽️ Dining Options</h4>
                            <div class="card-content interactive-card">
                                <ul style="list-style-type: none; padding-left: 0;">{dining_html}</ul>
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                
        # Create an enhanced interactive exploration section
        st.markdown('<h3 class="section-header">🌍 Explore & Discover</h3>', unsafe_allow_html=True)
        
        # Get user preferences for customization
        user_activities = st.session_state.preferences.get('preferred_activities', [])
        fitness_level = st.session_state.preferences.get('fitness_level', 'moderate')
        
        # Create tabs for different exploration modes
        img_tab, map_tab, plan_tab = st.tabs(["📸 Visual Preview", "🗺️ Interactive Locations", "📝 Planning Tools"])
        
        with img_tab:
            if day["images"]:
                # Enhanced image gallery with activity-based filtering
                st.markdown("### Choose Your View:")
                
                # Add view filters based on user interests
                view_options = ["All Photos"]
                if "Photography" in user_activities:
                    view_options.extend(["Best Photo Spots", "Golden Hour Views"])
                if "Hiking" in user_activities:
                    view_options.extend(["Trail Views", "Summit Panoramas"])
                if "Scenic Drives" in user_activities:
                    view_options.append("Road Trip Views")
                if "Beaches" in user_activities and any(keyword in day["location"].lower() for keyword in ["beach", "coast", "lofoten"]):
                    view_options.append("Coastal Scenes")
                
                selected_view = st.selectbox("Filter photos by:", view_options, key=f"view_{day['date']}")
                
                # Define captions and tags for better filtering
                enhanced_captions = {
                    "2025-08-02 (Saturday)": [
                        {"caption": "Sunset view from airplane approaching Norway", "tags": ["scenic", "golden_hour"], "activity": "Scenic Drives"}
                    ],
                    "2025-08-03 (Sunday)": [
                        {"caption": "Scenic coastal road in Lofoten Islands", "tags": ["road", "scenic"], "activity": "Scenic Drives"},
                        {"caption": "Mountain views along E10 to Lofoten", "tags": ["mountain", "road"], "activity": "Scenic Drives"}
                    ],
                    "2025-08-04 (Monday)": [
                        {"caption": "Haukland Beach with turquoise waters in summer", "tags": ["beach", "coast", "golden_hour"], "activity": "Beaches"},
                        {"caption": "Uttakleiv Beach and its iconic boulders", "tags": ["beach", "coast", "photo_spot"], "activity": "Photography"},
                        {"caption": "Panoramic view from Offersøykammen hike", "tags": ["hiking", "summit", "panorama"], "activity": "Hiking"}
                    ],
                    "2025-08-05 (Tuesday)": [
                        {"caption": "Viewpoint over Lofoten's dramatic mountains", "tags": ["hiking", "summit", "panorama"], "activity": "Hiking"},
                        {"caption": "Red rorbuer fishing cabins in Hamnøy", "tags": ["photo_spot", "village"], "activity": "Photography"},
                        {"caption": "Scenic Ramberg Beach with mountain backdrop", "tags": ["beach", "coast", "photo_spot"], "activity": "Beaches"}
                    ],
                    "2025-08-06 (Wednesday)": [
                        {"caption": "Traditional fishing village of Nusfjord in summer", "tags": ["village", "photo_spot"], "activity": "Photography"},
                        {"caption": "Sea eagle safari views in Lofoten", "tags": ["wildlife", "scenic"], "activity": "Wildlife Viewing"}
                    ],
                    "2025-08-07 (Thursday)": [
                        {"caption": "Henningsvær harbor village with mountains", "tags": ["village", "photo_spot"], "activity": "Photography"},
                        {"caption": "View from Fløya hiking trail in summer", "tags": ["hiking", "summit", "panorama"], "activity": "Hiking"}
                    ],
                    "2025-08-08 (Friday)": [
                        {"caption": "Bergen's colorful Bryggen Wharf in summer", "tags": ["photo_spot", "historic"], "activity": "Photography"},
                        {"caption": "Bergen harbor with boats in summer sunshine", "tags": ["scenic", "golden_hour"], "activity": "Scenic Drives"}
                    ],
                    "2025-08-09 (Saturday)": [
                        {"caption": "Bryggen Wharf historic buildings in summer", "tags": ["historic", "photo_spot"], "activity": "Photography"},
                        {"caption": "View from Mount Fløyen over Bergen", "tags": ["hiking", "summit", "panorama"], "activity": "Hiking"},
                        {"caption": "Bergen fish market in summer", "tags": ["food", "market"], "activity": "Food Tours"}
                    ],
                    "2025-08-10 (Sunday)": [
                        {"caption": "Summer view of Geirangerfjord UNESCO site", "tags": ["scenic", "photo_spot"], "activity": "Photography"},
                        {"caption": "Flydalsjuvet viewpoint over Geirangerfjord", "tags": ["hiking", "summit", "panorama"], "activity": "Hiking"}
                    ],
                    "2025-08-11 (Monday)": [
                        {"caption": "Seven Sisters waterfall in Geirangerfjord", "tags": ["scenic", "photo_spot"], "activity": "Photography"},
                        {"caption": "Cruise boat in summer on Geirangerfjord", "tags": ["scenic", "boat"], "activity": "Scenic Drives"}
                    ],
                    "2025-08-12 (Tuesday)": [
                        {"caption": "Canyoning adventure in Geirangerfjord", "tags": ["adventure", "extreme"], "activity": "Adventure Sports"},
                        {"caption": "Ålesund city view with art nouveau architecture", "tags": ["photo_spot", "historic"], "activity": "Photography"}
                    ],
                    "2025-08-13 (Wednesday)": [
                        {"caption": "Colorful wooden houses in Stavanger Old Town", "tags": ["photo_spot", "historic"], "activity": "Photography"},
                        {"caption": "Stavanger harbor in summer sunshine", "tags": ["scenic", "golden_hour"], "activity": "Scenic Drives"}
                    ],
                    "2025-08-14 (Thursday)": [
                        {"caption": "Kjeragbolten boulder wedged between cliffs", "tags": ["hiking", "extreme", "photo_spot"], "activity": "Adventure Sports"},
                        {"caption": "Summer hiking trail to Kjerag", "tags": ["hiking", "trail"], "activity": "Hiking"}
                    ],
                    "2025-08-15 (Friday)": [
                        {"caption": "Pulpit Rock (Preikestolen) in summer", "tags": ["hiking", "summit", "photo_spot"], "activity": "Hiking"},
                        {"caption": "View of Lysefjord from Pulpit Rock in August", "tags": ["hiking", "summit", "panorama"], "activity": "Hiking"}
                    ],
                    "2025-08-16 (Saturday)": [
                        {"caption": "Final view of Norwegian fjords and mountains", "tags": ["scenic", "panorama"], "activity": "Scenic Drives"}
                    ]
                }
                
                # Filter images based on selected view and user preferences
                day_image_data = enhanced_captions.get(day["date"], [])
                filtered_images = []
                
                for i, img_data in enumerate(day_image_data):
                    if i >= len(day["images"]):
                        break
                        
                    include_image = False
                    
                    if selected_view == "All Photos":
                        include_image = True
                    elif selected_view == "Best Photo Spots" and "photo_spot" in img_data.get("tags", []):
                        include_image = True
                    elif selected_view == "Golden Hour Views" and "golden_hour" in img_data.get("tags", []):
                        include_image = True
                    elif selected_view == "Trail Views" and "hiking" in img_data.get("tags", []):
                        include_image = True
                    elif selected_view == "Summit Panoramas" and "summit" in img_data.get("tags", []):
                        include_image = True
                    elif selected_view == "Road Trip Views" and "road" in img_data.get("tags", []):
                        include_image = True
                    elif selected_view == "Coastal Scenes" and any(tag in img_data.get("tags", []) for tag in ["beach", "coast"]):
                        include_image = True
                    
                    if include_image:
                        filtered_images.append((day["images"][i], img_data))
                
                # If no filtered images, show all
                if not filtered_images:
                    filtered_images = [(day["images"][i], day_image_data[i] if i < len(day_image_data) else {"caption": f"Norway Scene {i+1}", "tags": [], "activity": ""}) for i in range(len(day["images"]))]
                
                # Display filtered images with enhanced layout
                if len(filtered_images) == 1:
                    # Single image - large display with detailed info
                    img_path, img_data = filtered_images[0]
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        high_res_path = img_path.replace("Norway_gallery/", "Norway_gallery/original_backup/")
                        actual_path = high_res_path if os.path.exists(high_res_path) else img_path
                        high_quality_img = load_high_quality_image(actual_path)
                        
                        st.markdown('<div class="image-container large-image high-quality-image">', unsafe_allow_html=True)
                        st.image(high_quality_img, caption=img_data["caption"], width=600, output_format="PNG")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📋 Photo Details")
                        st.write(f"**Best for:** {img_data.get('activity', 'General viewing')}")
                        
                        if img_data.get('tags'):
                            st.write(f"**Tags:** {', '.join(img_data['tags'])}")
                        
                        # Add personalized photo tips
                        if "Photography" in user_activities:
                            if "golden_hour" in img_data.get("tags", []):
                                st.success("📸 Perfect for golden hour photography!")
                            if "photo_spot" in img_data.get("tags", []):
                                st.info("📷 This is a popular photography location")
                        
                        if "Hiking" in user_activities and "hiking" in img_data.get("tags", []):
                            if fitness_level in ["active", "very_active"]:
                                st.success("🥾 Great hiking destination for your fitness level!")
                            elif fitness_level == "light":
                                st.warning("⚠️ This may require moderate hiking")
                        
                        # Add practical info
                        st.markdown("### 💡 Planning Tips")
                        if "beach" in img_data.get("tags", []):
                            st.write("• Best visited during midday for warmest weather")
                            st.write("• Bring layers - coastal weather changes quickly")
                        elif "summit" in img_data.get("tags", []):
                            st.write("• Start early to avoid crowds")
                            st.write("• Check weather conditions before hiking")
                        elif "village" in img_data.get("tags", []):
                            st.write("• Great for morning or evening exploration")
                            st.write("• Many shops and cafes available")
                
                else:
                    # Multiple images - responsive grid
                    num_images = len(filtered_images)
                    if num_images > 3:
                        cols = st.columns([1, 1, 1])
                    elif num_images > 1:
                        cols = st.columns([1, 1])
                    else:
                        cols = st.columns([1])
                    
                    for i, (img_path, img_data) in enumerate(filtered_images):
                        col_idx = i % len(cols)
                        
                        high_res_path = img_path.replace("Norway_gallery/", "Norway_gallery/original_backup/")
                        actual_path = high_res_path if os.path.exists(high_res_path) else img_path
                        high_quality_img = load_high_quality_image(actual_path)
                        
                        with cols[col_idx]:
                            st.markdown('<div class="image-container high-quality-image">', unsafe_allow_html=True)
                            st.image(high_quality_img, caption=img_data["caption"], width=300, output_format="PNG")
                            
                            # Add quick activity match indicators
                            if img_data.get("activity") in user_activities:
                                st.markdown("⭐ *Matches your interests!*")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # Add personalized viewing suggestions
                if user_activities:
                    matching_activities = set()
                    for img_data in day_image_data:
                        if img_data.get("activity") in user_activities:
                            matching_activities.add(img_data.get("activity"))
                    
                    if matching_activities:
                        st.success(f"📸 This location has great photo opportunities for: {', '.join(matching_activities)}")
            
            else:
                st.info("📸 No photos available for this day yet. Check back later or explore the locations!")
        
        with map_tab:
            # Enhanced interactive location explorer
            st.markdown("### 🎯 Interactive Location Guide")
            
            if day["date"] in norway_locations:
                # Create enhanced location cards with detailed information
                for i, link in enumerate(norway_locations[day["date"]]):
                    # Extract and clean location name
                    match = re.search(r'place/([^/@]+)', link)
                    if match:
                        raw_name = match.group(1).replace('+', ' ').replace('_', ' ')
                        raw_name = re.sub(r'@[\d\.]+,[\d\.]+', '', raw_name)
                        name = ' '.join(word.capitalize() for word in raw_name.split())
                    else:
                        name = f"Location {i+1}"
                    
                    # Create enhanced location card
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        
                        with col1:
                            st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #4285F4, #34A853); 
                                           color: white; width: 50px; height: 50px; border-radius: 50%; 
                                           display: flex; align-items: center; justify-content: center; 
                                           font-weight: bold; font-size: 1.2rem; margin: auto;">
                                    {i+1}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"### 📍 {name}")
                            
                            # Add location-specific information based on user preferences
                            location_info = []
                            name_lower = name.lower()
                            
                            # Add activity-specific information
                            if "Hiking" in user_activities:
                                if any(keyword in name_lower for keyword in ["mountain", "rock", "cliff", "trail"]):
                                    if fitness_level in ["active", "very_active"]:
                                        location_info.append("🥾 Excellent hiking - matches your fitness level!")
                                    elif fitness_level == "moderate":
                                        location_info.append("🚶‍♂️ Moderate hiking available - take your time")
                                    else:
                                        location_info.append("⚠️ Challenging terrain - consider viewpoints instead")
                            
                            if "Photography" in user_activities:
                                if any(keyword in name_lower for keyword in ["fjord", "view", "falls", "bridge"]):
                                    location_info.append("📸 Outstanding photography opportunities!")
                                if "sunset" in name_lower or "sunrise" in name_lower:
                                    location_info.append("🌅 Perfect for golden hour photography")
                            
                            if "Food Tours" in user_activities:
                                if "market" in name_lower:
                                    location_info.append("🍽️ Great food exploration opportunities!")
                                elif "town" in name_lower or "city" in name_lower:
                                    location_info.append("🍴 Local dining scene worth exploring")
                            
                            if "Wildlife Viewing" in user_activities:
                                if any(keyword in name_lower for keyword in ["safari", "nature", "coast"]):
                                    location_info.append("🦅 Wildlife spotting opportunities!")
                            
                            # Add practical information
                            if "airport" in name_lower:
                                location_info.append("✈️ Transportation hub")
                            elif "beach" in name_lower:
                                location_info.append("🏖️ Coastal location - check weather")
                            elif "museum" in name_lower:
                                location_info.append("🏛️ Cultural attraction")
                            
                            # Display personalized information
                            if location_info:
                                for info in location_info:
                                    st.write(f"• {info}")
                            else:
                                st.write("📍 Scenic Norwegian destination")
                            
                            # Add estimated time and difficulty
                            if any(keyword in name_lower for keyword in ["rock", "mountain", "cliff"]):
                                if fitness_level == "very_active":
                                    st.write("⏱️ **Estimated time:** 4-6 hours")
                                    st.write("💪 **Difficulty:** Perfect for you!")
                                elif fitness_level == "active":
                                    st.write("⏱️ **Estimated time:** 5-7 hours")
                                    st.write("💪 **Difficulty:** Challenging but doable")
                                else:
                                    st.write("⏱️ **Estimated time:** Consider alternatives")
                                    st.write("💪 **Difficulty:** Very challenging")
                            elif "museum" in name_lower or "market" in name_lower:
                                st.write("⏱️ **Estimated time:** 1-3 hours")
                                st.write("💪 **Difficulty:** Easy walking")
                        
                        with col3:
                            st.markdown(f"""
                                <a href='{link}' target='_blank' style="text-decoration: none;">
                                    <div style="background: #4285F4; color: white; padding: 10px 15px; 
                                               border-radius: 25px; text-align: center; transition: all 0.3s;
                                               box-shadow: 0 2px 10px rgba(66, 133, 244, 0.3);">
                                        🗺️ Open Map
                                    </div>
                                </a>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
            
            else:
                st.info("🗺️ No specific locations mapped for this day. Enjoy the journey!")
        
        with plan_tab:
            # Simplified planning tools
            st.markdown("### 📝 Quick Planning")
            
            # Personal notes (simplified)
            st.markdown("#### 💭 Notes")
            current_note = st.session_state.personal_notes.get(day["date"], "")
            updated_note = st.text_area(
                "Add your thoughts or reminders:",
                value=current_note,
                height=80,
                key=f"notes_{day['date']}_simple",
                placeholder="Plans, reminders, must-dos..."
            )
            if updated_note != current_note:
                st.session_state.personal_notes[day["date"]] = updated_note
            
            # Favorites and completion (simplified)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⭐ Favorite")
                is_favorite = day["location"] in st.session_state.favorite_places
                if st.button("💖" if not is_favorite else "💔", 
                           key=f"plan_fav_{day['date']}", use_container_width=True):
                    if is_favorite:
                        st.session_state.favorite_places.discard(day["location"])
                    else:
                        st.session_state.favorite_places.add(day["location"])
                    st.rerun()
            
            with col2:
                st.markdown("#### ✅ Completed")
                is_visited = st.session_state.visited_status.get(day["date"], False)
                if st.button("✅" if not is_visited else "⏳", 
                           key=f"plan_visited_{day['date']}", use_container_width=True):
                    st.session_state.visited_status[day["date"]] = not is_visited
                    st.rerun()
        
        break
