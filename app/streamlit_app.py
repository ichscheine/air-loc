import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Air-Loc Explorer",
    page_icon="✈️",
    layout="wide"
)

# Data paths setup
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BRONZE_DIR = os.path.join(DATA_ROOT, "bronze")
SILVER_DIR = os.path.join(DATA_ROOT, "silver")
GOLD_DIR = os.path.join(DATA_ROOT, "gold")

# Load data
@st.cache_data
def load_data():
    # Find available cities by looking at gold data files
    city_files = glob.glob(os.path.join(GOLD_DIR, "*_neighborhood_features_gold.csv"))
    cities = [os.path.basename(f).split("_")[0] for f in city_files]
    
    data = {}
    for city in cities:
        city_data = {}
        
        # Load Gold layer (aggregated neighborhood insights)
        gold_path = os.path.join(GOLD_DIR, f"{city}_neighborhood_features_gold.csv")
        if os.path.exists(gold_path):
            city_data["gold"] = pd.read_csv(gold_path)
        
        # Load Silver layer (processed reviews)
        silver_path = os.path.join(SILVER_DIR, f"{city}_reviews_silver.parquet")
        if os.path.exists(silver_path):
            city_data["silver"] = pd.read_parquet(silver_path)
        
        # Load Bronze layer (raw data)
        city_data["bronze"] = {}
        # Reviews
        reviews_path = os.path.join(BRONZE_DIR, f"{city}_reviews.csv")
        if os.path.exists(reviews_path):
            city_data["bronze"]["reviews"] = pd.read_csv(reviews_path)
            
        # Listings
        listings_path = os.path.join(BRONZE_DIR, f"{city}_listings.csv")
        if os.path.exists(listings_path):
            city_data["bronze"]["listings"] = pd.read_csv(listings_path)
            
        # Neighborhoods
        neighborhoods_path = os.path.join(BRONZE_DIR, f"{city}_neighborhoods.geojson")
        if os.path.exists(neighborhoods_path):
            city_data["bronze"]["neighborhoods"] = gpd.read_file(neighborhoods_path)
            # Add centroid coordinates for mapping
            city_data["bronze"]["neighborhoods"]["latitude"] = city_data["bronze"]["neighborhoods"].geometry.centroid.y
            city_data["bronze"]["neighborhoods"]["longitude"] = city_data["bronze"]["neighborhoods"].geometry.centroid.x
            
        # POIs
        pois_path = os.path.join(BRONZE_DIR, f"{city}_pois.geojson")
        if os.path.exists(pois_path):
            city_data["bronze"]["pois"] = gpd.read_file(pois_path)
        
        data[city] = city_data
    
    return data, cities

# Main application
def main():
    st.title("🏙️ Air-Loc: Neighborhood Explorer")
    # st.write("Analyze Airbnb reviews and neighborhood characteristics")
    
    data, cities = load_data()
    
    if not cities:
        st.error("No data found. Please run the ETL pipeline first.")
        return
    
    # City selector
    city = st.sidebar.selectbox(
        "Select City", 
        options=cities,
        format_func=lambda x: x.title()  # Capitalize city names
    )
    city_data = data[city]
    
    # Check if we have data for this city
    if "gold" not in city_data:
        st.warning(f"No gold data available for {city}. Some insights will be limited.")
    if "silver" not in city_data:
        st.warning(f"No silver data available for {city}. Some visualizations will be limited.")
    if "bronze" not in city_data or not city_data["bronze"]:
        st.warning(f"No bronze data available for {city}. Raw data exploration will be limited.")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Neighborhood Insights", "Review Analysis", "Raw Data Explorer"])
    
    # Tab 1: Neighborhood Insights (Gold data)
    with tab1:
        # st.header(f"Neighborhood Insights for {city.title()}")
        
        if "gold" in city_data:
            gold_df = city_data["gold"]
            
            # Add Overall Statistics at the top of the page
            # Stats in a more visually appealing format with custom styling
            st.markdown(f"""
            <style>
            .stats-container {{
                display: flex;
                justify-content: space-between;
                background-color: #f8f9fa;
                padding: 10px 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #1e88e5;
            }}
            .stat-label {{
                font-size: 14px;
                color: #666;
            }}
            </style>

            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-value">{len(gold_df)}</div>
                    <div class="stat-label">Neighborhoods</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{gold_df['avg_sentiment'].mean():.2f}</div>
                    <div class="stat-label">Average Sentiment</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{gold_df['review_count'].sum():,}</div>
                    <div class="stat-label">Total Reviews</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Continue with your existing layout code
            # Use more compact layout
            main_cols = st.columns([5, 4])
            
            with main_cols[0]:
                # Sentiment map if we have neighborhood boundaries
                if "bronze" in city_data and "neighborhoods" in city_data["bronze"]:
                    neighborhoods = city_data["bronze"]["neighborhoods"]
                    
                    nbhd_col = next((col for col in neighborhoods.columns 
                                if any(x in col.lower() for x in ['name', 'neighbourhood', 'neighborhood', 'nbhd'])),
                                None)
                    
                    if nbhd_col:
                        neighborhoods = neighborhoods.rename(columns={nbhd_col: "neighborhood"})
                        merged = neighborhoods.merge(gold_df, on="neighborhood", how="left")
                        
                        # Create a more compact choropleth
                        fig, ax = plt.subplots(1, figsize=(4, 4), dpi=100)
                        merged.plot(column="avg_sentiment", 
                                cmap="RdYlGn", 
                                legend=True,
                                ax=ax,
                                legend_kwds={'shrink': 0.6})  # Make legend smaller
                        ax.set_axis_off()
                        # Use a smaller font for the title
                        ax.set_title(f"Average Sentiment by Neighborhood", fontsize=11)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Could not identify neighborhood name column.")
                else:
                    # Fallback bar chart
                    fig = px.bar(gold_df.sort_values("avg_sentiment"), 
                                x="neighborhood", y="avg_sentiment",
                                color="avg_sentiment", 
                                color_continuous_scale="RdYlGn",
                                title="Average Sentiment by Neighborhood")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with main_cols[1]:
                # Container for the tables and statistics
                with st.container():
                    # Create sub-columns to organize tables side by side
                    tables_col1, tables_col2 = st.columns(2)
                    
                    with tables_col1:
                        # Top neighborhoods
                        st.markdown("##### Top Neighborhoods")
                        top_df = gold_df.sort_values("avg_sentiment", ascending=False).head(5)
                        # Format numbers to save space
                        top_df["avg_sentiment"] = top_df["avg_sentiment"].map(lambda x: f"{x:.3f}")
                        # More compact table display
                        st.dataframe(
                            top_df[["neighborhood", "avg_sentiment", "review_count"]],
                            hide_index=True,
                            height=150,
                            use_container_width=True,  # Ensure table uses full container width
                            column_config={
                                "neighborhood": "Neighborhood",
                                "avg_sentiment": "Sentiment",
                                "review_count": "Reviews"
                            }
                        )
                    
                    with tables_col2:
                        # Bottom neighborhoods
                        st.markdown("##### Bottom Neighborhoods")
                        bottom_df = gold_df.sort_values("avg_sentiment").head(5)
                        bottom_df["avg_sentiment"] = bottom_df["avg_sentiment"].map(lambda x: f"{x:.3f}")
                        st.dataframe(
                            bottom_df[["neighborhood", "avg_sentiment", "review_count"]],
                            hide_index=True,
                            height=150,
                            use_container_width=True,  # Ensure table uses full container width
                            column_config={
                                "neighborhood": "Neighborhood",
                                "avg_sentiment": "Sentiment",
                                "review_count": "Reviews"
                            }
                        )
                
                # Add keyword summary if available
                if "top_keywords" in gold_df.columns:
                    # Create a container for keywords with custom styling
                    st.markdown("##### Neighborhood Keywords")
                    
                    # Show keywords for top neighborhoods
                    st.markdown("**Top Neighborhoods:**")
                    top_nbhds = gold_df.sort_values("avg_sentiment", ascending=False).head(5)
                    
                    # Use a more visually appealing format for the keywords
                    for _, row in top_nbhds.iterrows():
                        if pd.notna(row['top_keywords']) and row['top_keywords'] != "No distinct topics found":
                            # Create a styled box for each neighborhood's keywords
                            st.markdown(f"""
                            <div style="padding: 8px 15px; margin-bottom: 8px; border-left: 3px solid #28a745; background-color: #f8f9fa;">
                                <strong>{row['neighborhood']}</strong>: {row['top_keywords']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{row['neighborhood']}**: *No distinct topics found*")
                    
                    # Add some space
                    st.markdown("---")
                    
                    # Show keywords for bottom neighborhoods
                    st.markdown("**Bottom Neighborhoods:**")
                    bottom_nbhds = gold_df.sort_values("avg_sentiment").head(5)
                    
                    for _, row in bottom_nbhds.iterrows():
                        if pd.notna(row['top_keywords']) and row['top_keywords'] != "No distinct topics found":
                            # Create a styled box with different color for bottom neighborhoods
                            st.markdown(f"""
                            <div style="padding: 8px 15px; margin-bottom: 8px; border-left: 3px solid #dc3545; background-color: #f8f9fa;">
                                <strong>{row['neighborhood']}</strong>: {row['top_keywords']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{row['neighborhood']}**: *No distinct topics found*")
        else:
            st.info("Gold data not available for this city. Run the ETL pipeline to generate insights.")
            
    # Tab 2: Review Analysis (Silver data)
    with tab2:  
        # st.header(f"Review Analysis for {city.title()}")
        
        if "silver" in city_data:
            silver_df = city_data["silver"]
            
            # Use containers for better organization
            with st.container():
                # Top level visualizations - use full width for charts
                viz_cols = st.columns([1, 1])
                
                with viz_cols[0]:
                    # Sentiment distribution - simplify appearance
                    fig = px.histogram(silver_df, x="sentiment", 
                                title="Distribution of Review Sentiment",
                                labels={"sentiment": "Sentiment Score", "count": "Number of Reviews"},
                                color_discrete_sequence=["#3366cc"],
                                height=280)  # Limit height
                    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))  # Reduce margins
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_cols[1]:
                    # Neighborhood comparison - more compact
                    st.subheader("Explore Neighborhoods")
                    # Make the multiselect more compact
                    selected_neighborhoods = st.multiselect("Select neighborhoods to compare", 
                                                    options=sorted(silver_df["neighborhood"].unique()),
                                                    default=silver_df["neighborhood"].value_counts().head(3).index.tolist(),
                                                    max_selections=4)  # Limit selections to keep visualization clean
                    
                    if selected_neighborhoods:
                        filtered = silver_df[silver_df["neighborhood"].isin(selected_neighborhoods)]
                        fig = px.box(filtered, x="neighborhood", y="sentiment",
                                title="Sentiment Distribution by Neighborhood",
                                color="neighborhood",
                                height=250)  # Control the height
                        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False)  # Remove legend to save space
                        st.plotly_chart(fig, use_container_width=True)
            
            # Second row of visualizations
            with st.container():
                # Reviews over time and review examples
                viz_cols2 = st.columns([1, 1])
                
                with viz_cols2[0]:
                    if "date" in silver_df.columns:
                        # Convert to datetime if not already
                        silver_df["date"] = pd.to_datetime(silver_df["date"])
                        
                        # Group by month and calculate average sentiment
                        monthly = silver_df.set_index("date").resample("M")["sentiment"].mean().reset_index()
                        
                        fig = px.line(monthly, x="date", y="sentiment",
                                    title="Average Sentiment Over Time",
                                    labels={"sentiment": "Avg. Sentiment", "date": "Month"},
                                    height=280)  # Control height
                        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                
                with viz_cols2[1]:
                    # Review examples
                    st.subheader("Review Examples")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        positive = st.button("Most Positive", use_container_width=True)
                    with col2:
                        negative = st.button("Most Negative", use_container_width=True)
                    with col3:
                        random = st.button("Random", use_container_width=True)
                    
                    # Determine which reviews to show
                    if negative:
                        reviews = silver_df.sort_values("sentiment").head(3)
                        sentiment_type = "negative"
                    elif random:
                        reviews = silver_df.sample(3)
                        sentiment_type = "random"
                    else:  # Default to positive
                        reviews = silver_df.sort_values("sentiment", ascending=False).head(3)
                        sentiment_type = "positive"
                    
                    # Show fewer reviews with more compact layout
                    st.markdown(f"<p style='margin-bottom:5px'><em>Showing {sentiment_type} reviews:</em></p>", unsafe_allow_html=True)
                    
                    # Custom CSS for more compact reviews
                    st.markdown("""
                    <style>
                    .compact-review {
                        font-size: 0.85em;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #f8f9fa;
                        margin-bottom: 8px;
                    }
                    .review-meta {
                        color: #666;
                        font-size: 0.8em;
                    }
                    .review-text {
                        margin-top: 4px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display reviews in a more compact format
                    for _, row in reviews.iterrows():
                        neighborhood = row.get('neighborhood', 'Unknown')
                        sentiment = row['sentiment']
                        comment = row.get('comments', row.get('clean_comments', 'No comment'))
                        
                        # Color code based on sentiment
                        if sentiment > 0.2:
                            badge_color = "success"
                        elif sentiment < -0.2:
                            badge_color = "danger"
                        else:
                            badge_color = "secondary"
                        
                        st.markdown(f"""
                        <div class="compact-review">
                            <div class="review-meta">
                                <span class="badge bg-{badge_color}" style="padding:2px 6px;border-radius:10px;color:white;">
                                    {sentiment:.2f}
                                </span> &nbsp;
                                <strong>{neighborhood}</strong>
                            </div>
                            <div class="review-text">
                                {comment[:150]}{"..." if len(comment) > 150 else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Silver data not available for this city. Run the ETL pipeline to process reviews.")
        
    # Tab 3: Raw Data Explorer (Bronze data) - Update the POIs section to show categories first and add colors
    with tab3:
        # st.header(f"Raw Data Explorer for {city.title()}")
        
        if "bronze" in city_data and city_data["bronze"]:
            # Dropdown to select dataset
            data_options = list(city_data["bronze"].keys())
            selected_data = st.selectbox("Select dataset to explore", data_options)
            
            if selected_data:
                df = city_data["bronze"][selected_data]
                
                # Show first few rows
                st.subheader(f"{selected_data.title()} Data")
                st.write(f"Total rows: {len(df)}")
                st.write(df.head(10))
                
                # Show neighborhood on map if it has polygon coordinates
                if selected_data == "neighborhoods":
                    st.subheader("Neighborhood Map")
                    
                    # Use Plotly to show neighborhood boundaries instead of points
                    # Try to find the neighborhood name column
                    nbhd_col = next((col for col in df.columns 
                                if any(x in col.lower() for x in ['name', 'neighbourhood', 'neighborhood', 'nbhd'])),
                                None)
                    
                    # Create a choropleth map with neighborhood boundaries
                    fig = px.choropleth_mapbox(
                        df, 
                        geojson=df.__geo_interface__, 
                        locations=df.index,
                        color=nbhd_col if nbhd_col else None,  # Color by neighborhood if column exists
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        center={"lat": df.geometry.centroid.y.mean(), "lon": df.geometry.centroid.x.mean()},
                        mapbox_style="carto-positron",  # Light map background
                        zoom=10,
                        height=500,
                        opacity=0.7,  # Semi-transparent boundaries
                        labels={nbhd_col: "Neighborhood"} if nbhd_col else {},
                        hover_name=nbhd_col if nbhd_col else None
                    )
                    
                    # Improve the layout
                    fig.update_layout(
                        margin={"r":0, "t":0, "l":0, "b":0},
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_data == "pois" and "geometry" in df.columns:
                    # Extract center points for POIs and add coordinates
                    df["latitude"] = df.geometry.centroid.y
                    df["longitude"] = df.geometry.centroid.x
                    
                    # 1. FIRST SHOW POI CATEGORIES (switched order as requested)
                    if "amenity" in df.columns:
                        st.subheader("Top POI Categories")
                        # Get top categories
                        top_amenities = df["amenity"].value_counts().head(10)
                        
                        # Create a bar chart with a color palette
                        fig = px.bar(
                            top_amenities, 
                            # title="Top 10 POI Categories",
                            labels={"index": "Category", "value": "Count"},
                            color=top_amenities.index,  # Color by category
                            color_discrete_sequence=px.colors.qualitative.Bold,  # Use a colorful palette
                            height=350
                        )
                        fig.update_layout(showlegend=False)  # Hide legend to save space
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a dictionary mapping category to color for map
                        color_map = {cat: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)] 
                                for i, cat in enumerate(top_amenities.index)}
                        
                    # 2. THEN SHOW THE MAP with colored points
                    st.subheader("Points of Interest")
                    
                    # Filter to top categories and manageable number
                    sample_size = min(2000, len(df))
                    
                    # If we have amenity categories, color by category
                    if "amenity" in df.columns:
                        # Focus on top categories for better visualization
                        top_categories = top_amenities.index.tolist()
                        
                        # Create filtered dataset with only top categories (for cleaner map)
                        filtered_pois = df[df["amenity"].isin(top_categories)].sample(sample_size)
                        
                        # Add color column based on category
                        filtered_pois["color"] = filtered_pois["amenity"].map(color_map).fillna("#808080")
                        
                        # Create a custom Plotly scatter map
                        fig = px.scatter_mapbox(
                            filtered_pois,
                            lat="latitude",
                            lon="longitude",
                            color="amenity",
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            zoom=10,
                            height=500,
                            hover_name="amenity",
                            hover_data=["name"] if "name" in filtered_pois.columns else None,
                            title="Points of Interest by Category"
                        )
                        fig.update_layout(
                            mapbox_style="open-street-map",
                            margin={"r":0, "t":30, "l":0, "b":0}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add category filters for interactivity
                        st.markdown("##### Filter by category:")
                        category_cols = st.columns(5)  # Show categories in rows of 5
                        for i, category in enumerate(top_categories[:10]):  # Show top 10 max
                            with category_cols[i % 5]:
                                st.markdown(f"<span style='color:{color_map[category]};'>●</span> {category}", 
                                        unsafe_allow_html=True)
                        
                    else:
                        # If no categories, use random sample with default coloring
                        sample_pois = df.sample(sample_size)
                        # Use standard map
                        st.map(sample_pois[["latitude", "longitude"]])
        else:
            st.info("Bronze data not available for this city. Run the ETL pipeline to ingest raw data.")
            
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Air-Loc Explorer** helps you analyze neighborhood characteristics based on Airbnb reviews and POI data.
    
    Data is organized in the Medallion architecture:
    - **Bronze**: Raw data (reviews, listings, neighborhoods, POIs)
    - **Silver**: Cleaned and processed data with sentiment analysis
    - **Gold**: Aggregated insights by neighborhood
    """)

if __name__ == "__main__":
    main()