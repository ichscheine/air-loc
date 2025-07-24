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
/* High-quality image rendering enhancements */
.high-quality-image img {
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
    -webkit-backface-visibility: hidden;
    -ms-interpolation-mode: bicubic;
    transform: translateZ(0);
}

/* Weather display styling */
.weather-card {
    background: linear-gradient(to right, #f5f7fa, #e4e7eb);
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

/* Add styles for 3-day forecast in the sidebar */
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
    
    st.markdown('<div style="text-align: center; font-weight: bold; margin-bottom: 10px;">Norway Trip Packing List</div>', unsafe_allow_html=True)
    
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

# Now add the day selection after the checklist
st.sidebar.markdown('<div class="sidebar-header">Select a Day</div>', unsafe_allow_html=True)

for day in trip_data:
    if st.sidebar.button(day["date"], key=day["date"], help=f"View {day['location']}", 
                       use_container_width=True,
                       type="primary" if day["date"] == st.session_state.selected_date else "secondary"):
        st.session_state.selected_date = day["date"]

# Hidden functionality for developers - add a small discrete link at the bottom of the sidebar
with st.sidebar.expander("⚙️ Developer Options", expanded=False):
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

selected_date = st.session_state.selected_date

# Show details for selected date
for day in trip_data:
    if day["date"] == selected_date:
        # Create a cleaner navigation with compact controls
        # Find the current day index
        current_index = trip_data.index(day)
        
        # Create container for navigation with more compact layout
        nav_container = st.container()
        with nav_container:
            st.markdown('<div class="day-nav-buttons">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
        
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
                <h4>Itinerary Details</h4>
                <div class="card-content">{day['details']}</div>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Display activities if available
        if 'activities' in day and day['activities']:
            # Get place information with ratings for activities
            location_context = day["location"].split('→')[0].strip() if '→' in day["location"] else day["location"]
            activities_with_details = []
            
            for activity in day['activities']:
                # Get Google Maps URL and rating
                maps_url, rating = get_place_details(activity, location_context)
                rating_stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
                
                # Create HTML with rating and link
                activity_html = f"""<li>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 6px; border-radius: 6px;" class="clickable-item">
                        <div>{activity}</div>
                        <div>
                            <span style="color: #FFD700; margin-right: 5px;">{rating_stars}</span>
                            <span style="color: #666;">{rating}/5</span>
                            <a href="{maps_url}" target="_blank" style="margin-left: 10px; display: inline-flex; align-items: center; justify-content: center; background-color: #4285F4; color: white; width: 28px; height: 28px; border-radius: 50%; text-decoration: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s;">🗺️</a>
                        </div>
                    </div>
                </li>"""
                activities_with_details.append(activity_html)
            
            # Join all activities HTML
            activities_html = ''.join(activities_with_details)
            
            st.markdown(
                f"""<div class="content-card">
                    <h4>🚶‍♂️ Activities</h4>
                    <div class="card-content interactive-card">
                        <ul style="list-style-type: none; padding-left: 0;">{activities_html}</ul>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )
        
        # Display dining options if available
        if 'dining_options' in day and day['dining_options']:
            # Get place information with ratings for dining options
            location_context = day["location"].split('→')[0].strip() if '→' in day["location"] else day["location"]
            dining_with_details = []
            
            for dining in day['dining_options']:
                # Get Google Maps URL and rating
                maps_url, rating = get_place_details(dining, location_context)
                rating_stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
                
                # Create HTML with rating and link
                dining_html = f"""<li>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 6px; border-radius: 6px;" class="clickable-item">
                        <div>{dining}</div>
                        <div>
                            <span style="color: #FFD700; margin-right: 5px;">{rating_stars}</span>
                            <span style="color: #666;">{rating}/5</span>
                            <a href="{maps_url}" target="_blank" style="margin-left: 10px; display: inline-flex; align-items: center; justify-content: center; background-color: #4285F4; color: white; width: 28px; height: 28px; border-radius: 50%; text-decoration: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s;">🗺️</a>
                        </div>
                    </div>
                </li>"""
                dining_with_details.append(dining_html)
            
            # Join all dining HTML
            dining_html = ''.join(dining_with_details)
            
            st.markdown(
                f"""<div class="content-card">
                    <h4>🍽️ Dining Options</h4>
                    <div class="card-content interactive-card">
                        <ul style="list-style-type: none; padding-left: 0;">{dining_html}</ul>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )
                
        
        # Create a section for images and location links side by side
        st.markdown('<h3 class="section-header">📸 Images & Locations</h3>', unsafe_allow_html=True)
        
        # Create a two-column layout for images and location links
        image_col, link_col = st.columns([3, 1])  # 3:1 ratio to give more space to images
        
        with image_col:
            if day["images"]:
                # Define captions based on the day's activities
                captions = {
                    "2025-08-02 (Saturday)": ["Sunset view from airplane approaching Norway"],
                    "2025-08-03 (Sunday)": ["Scenic coastal road in Lofoten Islands", "Mountain views along E10 to Lofoten"],
                    "2025-08-04 (Monday)": ["Haukland Beach with turquoise waters in summer", "Uttakleiv Beach and its iconic boulders", "Panoramic view from Offersøykammen hike"],
                    "2025-08-05 (Tuesday)": ["Viewpoint over Lofoten's dramatic mountains", "Red rorbuer fishing cabins in Hamnøy", "Scenic Ramberg Beach with mountain backdrop"],
                    "2025-08-06 (Wednesday)": ["Traditional fishing village of Nusfjord in summer", "Sea eagle safari views in Lofoten"],
                    "2025-08-07 (Thursday)": ["Henningsvær harbor village with mountains", "View from Fløya hiking trail in summer"],
                    "2025-08-08 (Friday)": ["Bergen's colorful Bryggen Wharf in summer", "Bergen harbor with boats in summer sunshine"],
                    "2025-08-09 (Saturday)": ["Bryggen Wharf historic buildings in summer", "View from Mount Fløyen over Bergen", "Bergen fish market in summer"],
                    "2025-08-10 (Sunday)": ["Summer view of Geirangerfjord UNESCO site", "Flydalsjuvet viewpoint over Geirangerfjord"],
                    "2025-08-11 (Monday)": ["Seven Sisters waterfall in Geirangerfjord", "Cruise boat in summer on Geirangerfjord"],
                    "2025-08-12 (Tuesday)": ["Canyoning adventure in Geirangerfjord", "Ålesund city view with art nouveau architecture"],
                    "2025-08-13 (Wednesday)": ["Colorful wooden houses in Stavanger Old Town", "Stavanger harbor in summer sunshine"],
                    "2025-08-14 (Thursday)": ["Kjeragbolten boulder wedged between cliffs", "Summer hiking trail to Kjerag"],
                    "2025-08-15 (Friday)": ["Pulpit Rock (Preikestolen) in summer", "View of Lysefjord from Pulpit Rock in August"],
                    "2025-08-16 (Saturday)": ["Final view of Norwegian fjords and mountains"]
                }
                
                # Get captions for this day
                day_captions = captions.get(day["date"], [])
                
                # Use default captions if none are defined for this day
                if not day_captions or len(day_captions) < len(day["images"]):
                    day_captions = [f"Norway Scene {i+1}" for i in range(len(day["images"]))]
                
                # Use a consistent layout for better image sizing
                num_images = len(day["images"])
                if num_images > 3:
                    # Use 3 columns for 4+ images
                    cols = st.columns([1, 1, 1])
                elif num_images > 1:
                    # Use 2 columns for 2-3 images
                    cols = st.columns([1, 1])
                else:
                    # For single images, use a full-width column
                    cols = st.columns([1])  # Full-width for better image display
                
                for i, img_path in enumerate(day["images"]):
                    caption = day_captions[i] if i < len(day_captions) else f"Norway Scene {i+1}"
                    
                    # For multi-column layout, distribute across columns
                    if num_images > 1:
                        col_idx = i % len(cols)
                    else:
                        # For single image, always use the center column
                        col_idx = 0
                    
                    # Check if there's a high-resolution version in original_backup folder
                    high_res_path = img_path.replace("Norway_gallery/", "Norway_gallery/original_backup/")
                    actual_path = high_res_path if os.path.exists(high_res_path) else img_path
                    
                    # Load image with highest possible quality
                    high_quality_img = load_high_quality_image(actual_path)
                    
                    try:
                        with cols[col_idx]:
                            st.markdown('<div class="image-container large-image high-quality-image">', unsafe_allow_html=True)
                            # For single images, use explicit width for higher quality display
                            if num_images == 1:
                                # Use a larger fixed width for single images
                                st.image(high_quality_img, caption=caption, width=600, output_format="PNG")
                            else:
                                # For multiple images, use appropriate width for better quality
                                st.image(high_quality_img, caption=caption, width=300, output_format="PNG")
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not load image: {img_path}")
                        st.error(f"Error: {e}")
            else:
                # Display a more friendly message when no images are available
                st.markdown(
                    """<div class="content-card">
                        <h4>📸 No Photos Available</h4>
                        <div class="card-content" style="text-align:center;">
                            <p>No photos available for this day yet.</p>
                            <p>Check the location links for this destination.</p>
                        </div>
                    </div>""", 
                    unsafe_allow_html=True
                )
        
        # Show Google Maps links in the right column
        with link_col:
            if day["date"] in norway_locations:
                st.markdown(
                    """<div style="background-color: #f9f9f9; border-radius: 8px; padding: 10px;">
                        <h4 style="margin-top: 0;">📍 Locations</h4>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Extract location names from links
                location_names = []
                for link in norway_locations[day["date"]]:
                    # Extract location name from Google Maps URL
                    match = re.search(r'place/([^/@]+)', link)
                    if match:
                        # Clean up the name (replace + with space, etc)
                        name = match.group(1).replace('+', ' ').replace('_', ' ')
                        name = re.sub(r'@[\d\.]+,[\d\.]+', '', name)  # Remove coordinates
                        name = name.replace('/', ' - ')  # Replace slashes
                        name = ' '.join(word.capitalize() for word in name.split())  # Capitalize words
                    else:
                        name = f"Location {i+1}"
                    location_names.append(name)
                
                # Create a vertical list of location links
                for i, (link, name) in enumerate(zip(norway_locations[day["date"]], location_names)):
                    st.markdown(f"""<a href='{link}' target='_blank' class="location-link">
                        <div class="location-link-box">
                            <div class="location-link-number">{i+1}</div>
                            <div class="location-link-name">{name}</div>
                            <div style="margin-left: auto; color: #4285F4; opacity: 0.7;">
                                <span style="font-size: 16px;">🔗</span>
                            </div>
                        </div>
                    </a>""", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        break
