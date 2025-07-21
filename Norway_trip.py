import streamlit as st
import os
import json
import webbrowser
from pathlib import Path
import re
import subprocess
import platform

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
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-04 (Monday)",
        "location": "Haukland Beach – Uttakleiv Beach – Offersøykammen Hike",
        "details": """- Haukland Beach: white sands, turquoise water  
- Uttakleiv Beach: boulders and sunsets  
- Offersøykammen Hike: panoramic summit views
""",
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
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-07 (Thursday)",
        "location": "Henningsvær + Fløya",
        "details": """- Henningsvær: charming harbor, art galleries, football field  
- Fløya hike: spectacular harbor views
""",
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-08 (Friday)",
        "location": "Fly EVE → Bergen",
        "details": "Flight 3:40 PM – 5:30 PM to Bergen.",
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
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-11 (Monday)",
        "location": "Geirangerfjord Cruise",
        "details": """- Seven Sisters waterfall  
- The Suitor waterfall  
- UNESCO fjord scenery
""",
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-12 (Tuesday)",
        "location": "Geirangerfjord → Ålesund",
        "details": "Morning canyoning in Geirangerfjord. Drive back to Ålesund.",
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-13 (Wednesday)",
        "location": "Fly Ålesund → Stavanger",
        "details": """- Explore Stavanger Old Town  
- Colorful wooden houses and harbor views
""",
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-14 (Thursday)",
        "location": "Kjerag Hike",
        "details": """- Kjeragbolten hike: iconic boulder wedged between cliffs
""",
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-15 (Friday)",
        "location": "Pulpit Rock",
        "details": """- Hike to Preikestolen (Pulpit Rock) overlooking Lysefjord
""",
        "images": [],  # Will be populated after full trip_data is defined
    },
    {
        "date": "2025-08-16 (Saturday)",
        "location": "Return Home",
        "details": "Flight SVG → FRA → IAD. Depart 6:45 AM, arrive 1:30 PM.",
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
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Define the checklist data
checklist = {
    "Essentials": [
        "passport",
        "U.S. driver license",
        "phone, charger, and adapter(EU), powerbank"
    ],
    "Entertainment": [
        "selfie stick",
        "Kindle/books",
        "pocket-size game"
    ],
    "Protection/self-care": [
        "sunglasses",
        "sunscreen",
        "insect / mosquito repellent",
        "Slippers",
        "Personal items: water bottle; eye mask",
        "motion sickness relief (for boat tour and car sickness?)"
    ],
    "Hiking Gears": [
        "down jacket",
        "hiking shoes",
        "umbrella/or water resisitant rain coat"
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
    if st.sidebar.button(day["date"], key=day["date"], help=f"View {day['location']}"):
        st.session_state.selected_date = day["date"]

# Hidden functionality for developers - add a small discrete link at the bottom of the sidebar
with st.sidebar.expander("⚙️ Developer Options", expanded=False):
    # Add button to refresh image mapping
    if st.button("🔄 Refresh Image Mapping"):
        day_to_images = update_image_mapping()
        
        # Save the updated mapping
        with open(mapping_file, 'w') as f:
            json.dump(day_to_images, f, indent=2)
        
        st.success("Image mapping refreshed!")
        st.rerun()
    
    # Add button to open gallery folder
    if st.button("📁 Open Images Folder"):
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
                if col1.button("⬅️", help=f"Go to {prev_day['date']}"):
                    st.session_state.selected_date = prev_day["date"]
                    st.rerun()
            
            # Display header in center column with compact styling
            col2.markdown(f'<h2 class="location-header">📍 {day["location"]}</h2>', unsafe_allow_html=True)
            col2.markdown(f'<h3 class="date-subheader">{day["date"]}</h3>', unsafe_allow_html=True)
            
            # Next day button with improved styling
            if current_index < len(trip_data) - 1:
                next_day = trip_data[current_index + 1]
                if col3.button("➡️", help=f"Go to {next_day['date']}"):
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
        
        # Display details with improved formatting and more compact design
        st.markdown(
            f"""<div class="content-card">
                <h4>Itinerary Details</h4>
                <div class="card-content">{day['details']}</div>
            </div>""", 
            unsafe_allow_html=True
        )
        
        if day["images"]:
            st.markdown('<h3 class="section-header">📸 Images</h3>', unsafe_allow_html=True)
            
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
            if num_images > 4:
                # Use 3 columns for 5+ images
                cols = st.columns([1, 1, 1])
            elif num_images > 1:
                # Use 2 columns for 2-4 images
                cols = st.columns([1, 1])
            else:
                # For single images, use a wider center column that matches the circled image size
                cols = st.columns([1, 3, 1])  # Wider center column for better image display
            
            for i, img_path in enumerate(day["images"]):
                caption = day_captions[i] if i < len(day_captions) else f"Norway Scene {i+1}"
                
                # For multi-column layout, distribute across columns
                if num_images > 1:
                    col_idx = i % len(cols)
                else:
                    # For single image, always use the center column
                    col_idx = 1
                
                # Check if there's a high-resolution version in original_backup folder
                high_res_path = img_path.replace("Norway_gallery/", "Norway_gallery/original_backup/")
                actual_path = high_res_path if os.path.exists(high_res_path) else img_path
                
                try:
                    with cols[col_idx]:
                        st.markdown('<div class="image-container large-image">', unsafe_allow_html=True)
                        # For single images, use specific width to match circled image size
                        if num_images == 1:
                            st.image(actual_path, caption=caption, use_container_width=True)
                        else:
                            # For multiple images, use container width with additional CSS styling
                            st.image(actual_path, caption=caption, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load image: {img_path}")
                    st.error(f"Error: {e}")
        else:
            # Display a more friendly message in a card layout
            st.markdown(
                """<div class="content-card">
                    <h4>Photos</h4>
                    <div class="card-content" style="text-align:center;">
                        <p>No photos available for this day yet.</p>
                        <p>Check the Google Maps links below for this location.</p>
                    </div>
                </div>""", 
                unsafe_allow_html=True
            )
            
        # Always show Google Maps links for this location with improved styling
        if day["date"] in norway_locations:
            # Create a card-style container for Google Maps links
            st.markdown(
                """<div class="content-card">
                    <h4>📍 Location Links</h4>
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
            
            # Create a neat grid for location links
            cols = st.columns([1, 1])
            for i, (link, name) in enumerate(zip(norway_locations[day["date"]], location_names)):
                col_idx = i % 2
                with cols[col_idx]:
                    st.markdown(f"""<a href='{link}' target='_blank' class="location-link">
                        <div class="location-link-box">
                            <div class="location-link-number">{i+1}</div>
                            <div class="location-link-name">{name}</div>
                        </div>
                    </a>""", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        break
