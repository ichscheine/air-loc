# backend/pipeline.py

import os
import json
import requests
import pandas as pd
import geopandas as gpd
import argparse
import osmnx as ox
from shapely.geometry import Point, box
from textblob import TextBlob

# ── Directory Setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(PROJECT_ROOT, "data/bronze")   # Bronze layer
SILVER_DIR  = os.path.join(PROJECT_ROOT, "data/silver")   # Silver layer
GOLD_DIR    = os.path.join(PROJECT_ROOT, "data/gold")     # Gold layer
for d in (DATA_DIR, SILVER_DIR, GOLD_DIR):
    os.makedirs(d, exist_ok=True)


# ── Config Loader ─────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """Load cities configuration JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


# ── Bronze Layer (Ingestion) ──────────────────────────────────────────────────
def fetch_file(url: str, dest_path: str):
    resp = requests.get(url)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def fetch_pois_osmnx(city: str, bbox: list[float], tags: dict[str, bool] = None):
    """
    Uses OSMnx to fetch POIs within bbox for the given tags via polygon query.
    """
    west, south, east, north = bbox
    if tags is None:
        tags = {"amenity": True}
    # Create bounding polygon
    bounding_polygon = box(west, south, east, north)
    # Fetch all geometries matching tags within the polygon
    gdf = ox.features_from_polygon(bounding_polygon, tags)
    dest = os.path.join(DATA_DIR, f"{city}_pois.geojson")
    gdf.to_file(dest, driver="GeoJSON")
    print(f"[BRONZE] {city}: POIs ingested via OSMnx → {dest}")


def bronze_layer(city: str, cfg: dict):
    """Download reviews, neighborhoods, and POIs for one city."""
    # 1. Reviews
    reviews_url  = cfg["reviews_url"]
    reviews_path = os.path.join(DATA_DIR, f"{city}_reviews.csv")
    fetch_file(reviews_url, reviews_path)

    # 2. Listings (for lat/lon)
    listings_url  = cfg.get("listings_url") or reviews_url.replace("reviews.csv", "listings.csv")
    listings_path = os.path.join(DATA_DIR, f"{city}_listings.csv")
    fetch_file(listings_url, listings_path)

    # 3. Neighborhood boundaries GeoJSON
    nbhd_path = os.path.join(DATA_DIR, f"{city}_neighborhoods.geojson")
    fetch_file(cfg["neighborhoods_url"], nbhd_path)

    # 4. POIs via OSMnx
    fetch_pois_osmnx(city, cfg.get("bbox", []))

    print(f"[BRONZE] {city}: reviews, neighborhoods, POIs ingested")


# ── Silver Layer (Preprocessing) ──────────────────────────────────────────────
def silver_layer(city: str) -> gpd.GeoDataFrame:
    """Clean, sentiment, and spatially join reviews to neighborhoods."""
    reviews = pd.read_csv(os.path.join(DATA_DIR, f"{city}_reviews.csv"))
    listings = pd.read_csv(os.path.join(DATA_DIR, f"{city}_listings.csv"))
    neighborhoods = gpd.read_file(os.path.join(DATA_DIR, f"{city}_neighborhoods.geojson"))

    # rename and join to get lat/lon
    listings = listings.rename(columns={"id": "listing_id"})
    reviews  = reviews.merge(
        listings[["listing_id", "latitude", "longitude"]],
        on="listing_id",
        how="left"
    )
    
    # Clean & normalize
    reviews = reviews.dropna(subset=["comments","date","listing_id","latitude","longitude"])
    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews["clean_comments"] = reviews["comments"].str.replace(r"\s+"," ", regex=True).str.strip()

    # Sentiment
    reviews["sentiment"] = reviews["clean_comments"].apply(lambda txt: TextBlob(txt).sentiment.polarity)

    # Spatial join
    reviews["geometry"] = reviews.apply(lambda r: Point(r.longitude, r.latitude), axis=1)
    reviews_gdf = gpd.GeoDataFrame(reviews, geometry="geometry", crs=neighborhoods.crs)
    # Spatial join with the correct column name "neighbourhood" 
    joined = gpd.sjoin(reviews_gdf, neighborhoods[["neighbourhood", "geometry"]], 
                       how="left", predicate="within")
    
    # For consistency in the rest of the code, rename to "neighborhood" after joining
    joined = joined.rename(columns={"neighbourhood": "neighborhood"})
    
    # Save silver
    silver_path = os.path.join(SILVER_DIR, f"{city}_reviews_silver.parquet")
    joined.drop(columns="geometry").to_parquet(silver_path, index=False)
    print(f"[SILVER] {city}: saved silver data to {silver_path}")
    return joined


# ── Gold Layer (Feature Extraction) ───────────────────────────────────────────
def gold_layer(city: str, silver_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate neighborhood-level features: rating, sentiment, keywords."""
    print(f"[GOLD] {city}: Generating neighborhood-level features...")
    
    # Basic aggregations for sentiment and count
    agg = silver_gdf.groupby("neighborhood").agg(
        # avg_rating=("rating","mean"),
        avg_sentiment=("sentiment","mean"),
        review_count=("id","count"),
    ).reset_index()
    
    # Use BERTopic for more meaningful topic extraction
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        
        print(f"[GOLD] {city}: Extracting topics with BERTopic...")
        
        # Function to extract topics for each neighborhood
        def extract_neighborhood_topics(neighborhood_reviews, n_topics=3):
            # Skip if too few reviews
            if len(neighborhood_reviews) < 10:
                return "Not enough reviews for topic modeling"
            
            # Combine all comments into a list
            documents = neighborhood_reviews["clean_comments"].tolist()
            
            # Configure BERTopic for better results on short texts
            # 1. Use custom CountVectorizer to keep important terms
            vectorizer = CountVectorizer(
                stop_words="english",  # Remove English stopwords
                min_df=5,              # Term must appear in at least 5 documents
                ngram_range=(1, 2)     # Allow bigrams for more context
            )
            
            # 2. Create BERTopic model with custom settings
            topic_model = BERTopic(
                vectorizer_model=vectorizer,
                min_topic_size=5,      # Require at least 5 documents per topic
                verbose=True,
                calculate_probabilities=False,  # Speed up computation
                n_gram_range=(1, 2)    # Consider bigrams in representation
            )
            
            # Fit the model and transform documents
            topics, _ = topic_model.fit_transform(documents)
            
            # Get the top n topics
            topic_info = topic_model.get_topic_info()
            top_topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1][:n_topics]
            
            # Get keywords for each topic
            topic_keywords = []
            for topic_id in top_topic_ids:
                if topic_id == -1:  # Skip outlier topic
                    continue
                words = [word for word, _ in topic_model.get_topic(topic_id)][:5]
                topic_keywords.append(" ".join(words))
            
            # Return comma-separated topic phrases
            return " | ".join(topic_keywords) if topic_keywords else "No distinct topics found"
        
        # Group by neighborhood and extract topics
        neighborhood_topics = {}
        for neighborhood, group in silver_gdf.groupby("neighborhood"):
            neighborhood_topics[neighborhood] = extract_neighborhood_topics(group)
        
        # Add topics to the aggregated dataframe
        agg["top_keywords"] = agg["neighborhood"].map(neighborhood_topics)
        
    except ImportError:
        print("[WARN] BERTopic not installed. Falling back to simple keyword extraction.")
        # Fall back to the original method if BERTopic is not available
        
        def top_words(texts, n=5):
            # Extended stopwords list
            stopwords = {'the', 'and', 'to', 'was', 'a', 'is', 'in', 'it', 'for', 'of', 'with', 
                        'this', 'that', 'we', 'are', 'i', 'my', 'from', 'on', 'at', 'our', 'were',
                        'had', 'has', 'have', 'be', 'so', 'but', 'very', 'not', 'by', 'an', 'as',
                        'there', 'they', 'you', 'would', 'could', 'just', 'all', 'some', 'one',
                        'what', 'when', 'who', 'how', 'where', 'which', 'or', 'if', 'their',
                        'your', 'us', 'about', 'out', 'up', 'down', 'no', 'can', 'will',
                        'than', 'then', 'its', "it's", 'his', 'her', 'he', 'she', 'place',
                        'stay', 'seattle', 'apartment', 'house', 'room', 'airbnb', 'stayed',
                        'home', 'get', 'also', 'really', 'great', 'good', 'nice', 'clean',
                        'location', 'would', 'time', 'like', 'well', 'stay', 'going', 'stayed'}
            
            # Join all texts, convert to lowercase, and split into words
            words = " ".join(texts).lower().split()
            
            # Filter out stopwords and very short words
            filtered_words = [word for word in words 
                             if word not in stopwords and len(word) > 2]
            
            # Return top meaningful words
            return ", ".join(pd.Series(filtered_words).value_counts().head(n).index)
        
        keywords = silver_gdf.groupby("neighborhood")["clean_comments"].apply(lambda tx: top_words(tx,5))
        agg["top_keywords"] = keywords.values

    # Save gold layer
    gold_path = os.path.join(GOLD_DIR, f"{city}_neighborhood_features_gold.csv")
    agg.to_csv(gold_path, index=False)
    print(f"[GOLD]  {city}: saved gold features to {gold_path}")
    return agg

# ── Orchestration ─────────────────────────────────────────────────────────────
# Add a new function to load the silver data from parquet
def load_silver_data(city: str) -> gpd.GeoDataFrame:
    """Load the silver data from parquet file."""
    silver_path = os.path.join(SILVER_DIR, f"{city}_reviews_silver.parquet")
    if not os.path.exists(silver_path):
        raise FileNotFoundError(f"Silver data for {city} not found at {silver_path}")
    
    # Load the parquet file
    silver_df = pd.read_parquet(silver_path)
    
    # Convert to GeoDataFrame if needed
    if "latitude" in silver_df.columns and "longitude" in silver_df.columns:
        silver_df["geometry"] = gpd.points_from_xy(silver_df.longitude, silver_df.latitude)
        silver_gdf = gpd.GeoDataFrame(silver_df, geometry="geometry", crs="EPSG:4326")
        return silver_gdf
    
    return silver_df

# Update the orchestration function to support regenerating just gold layer
def run_pipeline(cities: list[str], config: dict, layers: list[str] = ["bronze", "silver", "gold"]):
    for city in cities:
        if city not in config:
            print(f"[WARN] {city} not in config → skipping")
            continue
        print(f"\n=== Processing: {city} ===")
        
        # Run each requested layer
        if "bronze" in layers:
            bronze_layer(city, config[city])
        
        if "silver" in layers:
            silver = silver_layer(city)
        elif "gold" in layers:
            # Load silver data if we're skipping the silver layer but need it for gold
            try:
                silver = load_silver_data(city)
                print(f"[INFO] {city}: loaded existing silver data")
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                print(f"[INFO] {city}: skipping gold layer as silver data is missing")
                continue
        
        if "gold" in layers:
            gold = gold_layer(city, silver)
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Airbnb Neighborhood Explorer ETL pipeline")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config", "cities_config.json"), help="Path to cities_config.json")
    parser.add_argument("--cities", nargs="+", default=None, help="List of city keys to process (from config)")
    parser.add_argument("--layers", nargs="+", default=["bronze", "silver", "gold"], help="Layers to process (bronze, silver, gold)")
    args = parser.parse_args()

    city_config = load_config(args.config)
    cities_to_run = args.cities or list(city_config.keys())
    run_pipeline(cities_to_run, city_config, args.layers)

