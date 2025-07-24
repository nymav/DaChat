import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load raw CSV files"""
    data_dir = Path("data/raw")
    
    try:
        listings = pd.read_csv(data_dir / "listings.csv")
        calendar = pd.read_csv(data_dir / "calendar.csv.gz")
        reviews = pd.read_csv(data_dir / "reviews.csv")
        
        logger.info(f"Loaded {len(listings)} listings, {len(calendar)} calendar entries, {len(reviews)} reviews")
        return listings, calendar, reviews
    
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_listings(listings):
    """Clean listings data"""
    # Remove duplicates
    listings = listings.drop_duplicates(subset=['id'])
    
    # Handle missing values
    listings['name'] = listings['name'].fillna('Unknown')
    listings['description'] = listings['description'].fillna('')
    
    # Clean price column (remove $ and convert to float)
    if 'price' in listings.columns:
        listings['price'] = listings['price'].str.replace('$', '').str.replace(',', '')
        listings['price'] = pd.to_numeric(listings['price'], errors='coerce')
        listings = listings.dropna(subset=['price'])
    
    # Clean numeric columns
    numeric_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    for col in numeric_cols:
        if col in listings.columns:
            listings[col] = pd.to_numeric(listings[col], errors='coerce')
            listings[col] = listings[col].fillna(0)
    
    logger.info(f"Cleaned listings: {len(listings)} records remaining")
    return listings

def clean_calendar(calendar):
    """Clean calendar data"""
    # Convert date column
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    # Clean price column
    if 'price' in calendar.columns:
        calendar['price'] = calendar['price'].str.replace('$', '').str.replace(',', '')
        calendar['price'] = pd.to_numeric(calendar['price'], errors='coerce')
    
    # Convert available to boolean
    calendar['available'] = calendar['available'].map({'t': True, 'f': False})
    
    logger.info(f"Cleaned calendar: {len(calendar)} records")
    return calendar

def aggregate_reviews(reviews):
    """Aggregate reviews by listing"""
    if reviews.empty:
        return pd.DataFrame(columns=['listing_id', 'review_count', 'avg_rating'])
    
    # Group by listing and calculate metrics
    review_stats = reviews.groupby('listing_id').agg({
        'id': 'count',  # review count
        'date': 'max'   # last review date
    }).reset_index()
    
    review_stats.columns = ['listing_id', 'review_count', 'last_review']
    
    logger.info(f"Aggregated reviews for {len(review_stats)} listings")
    return review_stats

def merge_data(listings, calendar, review_stats):
    """Merge all datasets"""
    # Start with listings as base
    merged = listings.copy()
    
    # Add review statistics
    if not review_stats.empty:
        merged = merged.merge(review_stats, left_on='id', right_on='listing_id', how='left')
        merged = merged.drop('listing_id', axis=1)
    else:
        merged['review_count'] = 0
        merged['last_review'] = None
    
    # Add calendar statistics (availability rate, average price)
    if not calendar.empty:
        calendar_stats = calendar.groupby('listing_id').agg({
            'available': 'mean',  # availability rate
            'price': 'mean'       # average price
        }).reset_index()
        calendar_stats.columns = ['listing_id', 'availability_rate', 'calendar_avg_price']
        
        merged = merged.merge(calendar_stats, left_on='id', right_on='listing_id', how='left')
        merged = merged.drop('listing_id', axis=1)
    
    # Fill missing values
    merged['review_count'] = merged['review_count'].fillna(0)
    merged['availability_rate'] = merged['availability_rate'].fillna(0)
    
    logger.info(f"Final merged dataset: {len(merged)} records")
    return merged

def save_processed_data(data):
    """Save processed data"""
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "merged_data.csv"
    data.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path}")
    return output_path

def main():
    """Main processing pipeline"""
    try:
        # Load data
        listings, calendar, reviews = load_data()
        
        # Clean data
        listings_clean = clean_listings(listings)
        calendar_clean = clean_calendar(calendar)
        review_stats = aggregate_reviews(reviews)
        
        # Merge data
        merged_data = merge_data(listings_clean, calendar_clean, review_stats)
        
        # Save processed data
        save_processed_data(merged_data)
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise

if __name__ == "__main__":
    main()