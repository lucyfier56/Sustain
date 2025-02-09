import gdelt
import os
import datetime
from collections import Counter
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from itertools import product
import time
import warnings
import logging
from tqdm import tqdm
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gdelt_processing.log'),
        logging.StreamHandler()
    ]
)

# Filter GDELT warnings
warnings.filterwarnings('ignore', message='GDELT did not return data for date time.*')

class Config:
    N_COMPANIES = 200
    START_DATE = "2024-01-01"
    END_DATE = "2025-02-06"  # 3 days only
    OUTPUT_DIR = "gdelt_data_3days"
    OUTPUT_FILENAME = f"top_{N_COMPANIES}_companies_365days_2025.csv"
    BATCH_SIZE = 10  # Process all 3 days without pausing
    PAUSE_TIME = 30  # Reduced pause time
    MAX_THREADS = 5  # Number of threads to use

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def get_russell_companies(n=Config.N_COMPANIES, sector=None):
    """Get the top n Russell 1000 companies"""
    def rm_class(s):
        class_types = ["CLASS", "SERIES", "REIT", "SHS"]
        share_class = [" ".join(x) + r"( |$)" for x in product(class_types,
                       ["A", "B", "C", "I"])]
        for sc in share_class:
          s = re.sub(sc, "", s).strip()
        return s

    url = ("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
           "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund")
    
    logging.info("Downloading Russell 1000 companies list...")
    try:
        russell_1000 = pd.read_csv(url, skiprows=range(9), header=0,
                                   usecols=["Name", "Sector"])
        russell_1000["Name"] = russell_1000.Name.astype(str).apply(rm_class)
        russell_1000.drop_duplicates(inplace=True)

        if sector:
            russell_1000 = russell_1000[russell_1000.Sector == sector]
        
        companies = russell_1000.iloc[:n].Name.to_list()
        logging.info(f"Successfully retrieved {len(companies)} companies")
        return companies
    except Exception as e:
        logging.error(f"Error retrieving companies: {str(e)}")
        raise

def make_aliases(c):
    """Make a list of aliases for a given company"""
    c = c.lower()
    if len(c) > 3 and c[-4] == ".":
        a = c
        c = c.rsplit(".", 1)[0].replace(".", " ")
        aliases = set([c, a])
    else:
        aliases = set([c])

    if len(c.split()[-1]) == 1:
        c = c.rsplit(" ", 1)[0]
        aliases.add(c)

    endings = ["inc", "corp", "plc", "reit", "co", "cor", "group", "company",
               "trust", "energy", "international", "of america", "pharmaceuticals",
               "clas", "in", "nv", "sa", "re"]
    
    for _ in range(3):
        aliases.update([a.rsplit(" ", 1)[0] for a in aliases if
                        any([a.endswith(" " + e) for e in endings])])
        c = c.rsplit(" ", 1)[0] if any([c.endswith(" " + e) for e in endings]) else c

    aliases.update([a.replace("-", "") for a in aliases] +
                   [a.replace("-", " ") for a in aliases])
    c = c.replace("-", " ")
    aliases.update([a.replace(" & ", " and ") for a in aliases])

    return {c: list(aliases)}

class Fields:
    keep = ["DATE", "SourceCommonName", "DocumentIdentifier", "Themes",
            "Organizations", "V2Tone"]
    tone = ["Tone", "PositiveTone", "NegativeTone", "Polarity",
            "ActivityDensity", "SelfDensity", "WordCount"]
    organizations = {k: v for d in [make_aliases(c) for c in get_russell_companies()] 
                    for k, v in d.items()}

class GDELTProcessor:
    def __init__(self):
        self.gdelt = gdelt.gdelt(version=2)
        self.total_processed = 0
        self.total_skipped = 0
        self.records_processed = 0
        
    def process_day(self, date):
        """Process one day of GDELT data"""
        try:
            logging.info(f"Processing {date}...")
            # Get GDELT data
            df = self.gdelt.Search(date, table="gkg", coverage=True, output="df")
            
            if df is None or df.empty:
                self.total_skipped += 1
                logging.info(f"No data available for {date}")
                return None
                
            df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S")
            
            # Process the data
            df = df[Fields.keep].copy()
            df["Themes"] = df["Themes"].fillna("").str.split(";")
            df["Organizations"] = df["Organizations"].fillna("").str.split(";")
            
            # Process tone
            tone_df = df["V2Tone"].str.split(",", expand=True)
            tone_df = tone_df.apply(pd.to_numeric, errors='coerce')
            tone_df.columns = Fields.tone
            df = pd.concat([df, tone_df], axis=1)
            df = df.drop("V2Tone", axis=1)
            
            # Create ESG columns
            df["E"] = df["Themes"].apply(lambda x: any("ENV" in t.split("_") for t in x if isinstance(t, str)))
            df["S"] = df["Themes"].apply(lambda x: any("UNGP" in t.split("_") for t in x if isinstance(t, str)))
            df["G"] = df["Themes"].apply(lambda x: any("ECON" in t.split("_") for t in x if isinstance(t, str)))
            
            # Process organizations
            df = df.explode("Organizations")
            df["Organization"] = df["Organizations"].str.lower()
            
            # Map organizations
            org_map = {alias: key for key, aliases in Fields.organizations.items() 
                      for alias in aliases}
            df["Organization"] = df["Organization"].map(org_map)
            df = df[df["Organization"].notna()]
            
            self.total_processed += 1
            self.records_processed += len(df)
            logging.info(f"Successfully processed {date} with {len(df)} records")
            return df
            
        except Exception as e:
            self.total_skipped += 1
            logging.error(f"Error processing {date}: {str(e)}")
            return None

def save_summary_statistics(processor, final_df, output_dir):
    """Save summary statistics to a file"""
    summary_path = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Processing Summary for {Config.OUTPUT_FILENAME}\n")
        f.write(f"Generated on: {datetime.datetime.now()}\n\n")
        f.write(f"Days processed successfully: {processor.total_processed}\n")
        f.write(f"Days skipped/failed: {processor.total_skipped}\n")
        f.write(f"Total records: {len(final_df):,d}\n")
        f.write(f"Unique companies: {final_df['Organization'].nunique():,d}\n")
        f.write(f"Date range: {final_df['DATE'].min()} to {final_df['DATE'].max()}\n\n")
        
        # ESG statistics
        f.write("ESG Statistics:\n")
        f.write(f"Environmental mentions: {final_df['E'].sum():,d}\n")
        f.write(f"Social mentions: {final_df['S'].sum():,d}\n")
        f.write(f"Governance mentions: {final_df['G'].sum():,d}\n\n")
        
        # Company statistics
        f.write("Records per company:\n")
        company_counts = final_df['Organization'].value_counts()
        for company, count in company_counts.items():
            f.write(f"{company}: {count:,d}\n")

def main():
    try:
        # Create output directory
        ensure_directory(Config.OUTPUT_DIR)
        output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILENAME)
        
        logging.info(f"Starting processing for {Config.N_COMPANIES} companies from {Config.START_DATE} to {Config.END_DATE}")
        
        # Initialize processor
        processor = GDELTProcessor()
        
        # Process data day by day
        all_data = []
        dates = pd.date_range(Config.START_DATE, Config.END_DATE)
        
        # Process with multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_THREADS) as executor:
            future_to_date = {executor.submit(processor.process_day, date.strftime("%Y-%m-%d")): date for date in dates}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(dates), desc="Processing dates"):
                date = future_to_date[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logging.error(f"Error processing {date}: {str(e)}")
        
        # Combine and save data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            save_cols = ["DATE", "SourceCommonName", "DocumentIdentifier", "E", "S", "G",
                         "Organization", "Tone", "PositiveTone", "NegativeTone",
                         "Polarity", "ActivityDensity", "SelfDensity", "WordCount"]
            
            final_df = final_df[save_cols]
            
            logging.info(f"Saving {len(final_df):,d} records to {output_path}")
            final_df.to_csv(output_path, index=False)
            
            save_summary_statistics(processor, final_df, Config.OUTPUT_DIR)
            
            logging.info("Processing completed successfully!")
            
        else:
            logging.error("No data was processed successfully")
            
    except Exception as e:
        logging.error(f"Fatal error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
