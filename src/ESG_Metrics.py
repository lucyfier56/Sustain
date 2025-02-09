import os
from pprint import pprint
import pandas as pd
import numpy as np
import datetime
import sys

class Fields:
    n_orgs = 100

def show_df_rounded(df, places=4, rows=20):
    """Display DataFrame with rounded numbers"""
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(places)
    print(df_display.head(rows))

def forward_fill(df, sort_col, fill_col):
    """Forward fill missing values in a pandas DataFrame"""
    df = df.sort_values(sort_col)
    df[fill_col] = df[fill_col].ffill()
    return df

def fill_dates(df, min_date, max_date, date_col="date"):
    """Add all days in a date range then forward fill"""
    date_range = pd.date_range(min_date, max_date)
    date_df = pd.DataFrame({date_col: date_range})
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = pd.merge(date_df, df, on=date_col, how='left')
    
    for col in [c for c in df.columns if c != date_col]:
        df = forward_fill(df, date_col, col)
    
    return df

def daily_tone(filtered_df, name):
    """Calculate daily tone"""
    colname = f"{name.replace(' ', '_')}_tone"
    
    # Create a copy of the dataframe
    df = filtered_df.copy()
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Group by date and calculate tone
    grouped = df.groupby(df['DATE'].dt.date).agg({
        'Tone': lambda x: (x * df.loc[x.index, 'WordCount']).sum(),
        'WordCount': 'sum'
    })
    
    # Calculate weighted average
    tone_df = pd.DataFrame({
        'date': grouped.index,
        colname: grouped['Tone'] / grouped['WordCount']
    }).reset_index(drop=True)
    
    return tone_df

def subtract_cols(df, col1, col2):
    """Subtract two columns and rename the result"""
    new_col = col1.replace("_tone", "_diff")
    df = df.copy()
    df[new_col] = df[col1] - df[col2]
    df = df.drop(columns=[col1])
    return df

def get_col_avgs(df):
    """Get averages of numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].mean().to_dict()

def make_tables(data_path, start_date, end_date, output_dir):
    """Main function to process data and create tables"""
    # Load data
    try:
        data = pd.read_csv(data_path)
        print("Data Loaded!")
    except Exception as e:
        print(f"Could not load data file: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all organizations
    print("Finding all Organizations")
    organizations = data['Organization'].unique()
    
    # Get the overall tone
    print("Calculating Tones Over Time")
    overall_tone = daily_tone(data, "industry")
    
    # Calculate ESG tones
    esg_tones = {}
    for L in ["E", "S", "G"]:
        mask = data[L] == True
        if mask.any():
            esg_tones[L] = daily_tone(data[mask], "industry")
    
    # Process organizations
    print("Processing organizations...")
    total_orgs = len(organizations)
    
    for i, org in enumerate(organizations):
        if i % max(1, total_orgs // 10) == 0:
            print(f"{(i * 100) // total_orgs}% complete")
            
        tone_label = f"{org.replace(' ', '_')}_tone"
        
        # Overall organization tone
        org_mask = data['Organization'] == org
        if org_mask.any():
            org_data = data[org_mask].copy()
            org_tone = daily_tone(org_data, org)
            
            overall_tone = subtract_cols(
                pd.merge(overall_tone, org_tone, on="date", how="left"),
                tone_label, "industry_tone"
            )
            
            # ESG tones
            for L in ["E", "S", "G"]:
                if L in esg_tones:
                    esg_mask = org_data[L] == True
                    if esg_mask.any():
                        esg_org_tone = daily_tone(org_data[esg_mask], org)
                        esg_tones[L] = subtract_cols(
                            pd.merge(esg_tones[L], esg_org_tone, on="date", how="left"),
                            tone_label, "industry_tone"
                        )
    
    # Calculate scores
    print("Computing Overall Scores")
    scores = {}
    overall_scores = get_col_avgs(overall_tone)
    esg_scores = {L: get_col_avgs(tdf) for L, tdf in esg_tones.items()}
    
    for org in organizations:
        diff_label = f"{org.replace(' ', '_')}_diff"
        scores[org] = {
            L: esg_scores.get(L, {}).get(diff_label, np.nan) 
            for L in ["E", "S", "G"]
        }
        scores[org]["T"] = overall_scores.get(diff_label, np.nan)
    
    # Save tables
    print("Saving Tables")
    
    # Overall ESG
    print("    Daily Overall ESG")
    overall_tone.to_csv(os.path.join(output_dir, "overall_daily_esg_scores.csv"), index=False)
    
    # E, S, and G scores
    for L, tdf in esg_tones.items():
        print(f"    Daily {L}")
        tdf.to_csv(os.path.join(output_dir, f"daily_{L}_score.csv"), index=False)
    
    # Average scores
    print("    Average Scores")
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(os.path.join(output_dir, "average_esg_scores.csv"))
    
    print("DONE!")

def process_date_range(date_str):
    """Convert date string to datetime object"""
    try:
        return pd.to_datetime(date_str).date()
    except Exception as e:
        print(f"Error processing date {date_str}: {e}")
        return None

if __name__ == "__main__":
    # Configuration
    INPUT_DATA_PATH = "/home/rudra-panda/Desktop/ESG/ESG_AI/gdelt_data_3days/top_200_companies_3days_2025.csv"  # Replace with your input file path
    OUTPUT_DIR = "/home/rudra-panda/Desktop/ESG/ESG_AI/gdelt_data_3days"      # Replace with your output directory
    START_DATE = "2025-01-01"                    # Replace with your start date
    END_DATE = "2025-01-03"                      # Replace with your end date

    # Validate dates
    start = process_date_range(START_DATE)
    end = process_date_range(END_DATE)
    
    if start is None or end is None:
        print("Invalid date format. Please use YYYY-MM-DD format.")
        sys.exit(1)
    
    if start > end:
        print("Start date must be before end date.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run the analysis
    try:
        make_tables(
            data_path=INPUT_DATA_PATH,
            start_date=start,
            end_date=end,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        sys.exit(1)
