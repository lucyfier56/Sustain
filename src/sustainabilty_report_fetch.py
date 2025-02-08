from googlesearch import search
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
from fake_useragent import UserAgent
import time
import re
from datetime import datetime
import os

class SustainabilityReportFinder:
    def __init__(self):
        self.ua = UserAgent()
        self.setup_logging()
        self.company_domains = {}
        self.output_dir = 'sustainability_reports'
        self.create_output_directory()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='sustainability_report_finder.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def create_output_directory(self):
        """Create main output directory and company-specific subdirectories"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created main output directory: {self.output_dir}")

    def create_company_directory(self, company_name):
        """Create company-specific directory"""
        # Clean company name for directory
        company_dir = re.sub(r'[^\w\-_]', '_', company_name)
        company_path = os.path.join(self.output_dir, company_dir)
        if not os.path.exists(company_path):
            os.makedirs(company_path)
            logging.info(f"Created company directory: {company_path}")
        return company_path

    def download_pdf(self, url, company_name):
        """
        Download and save PDF file
        """
        try:
            # Create company directory
            company_dir = self.create_company_directory(company_name)
            
            # Generate filename from URL and date
            url_filename = url.split('/')[-1]
            base_name = os.path.splitext(url_filename)[0]
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"{base_name}_{date_str}.pdf"
            filepath = os.path.join(company_dir, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                logging.info(f"File already exists: {filepath}")
                return filepath

            # Download the file
            headers = {'User-Agent': self.ua.random}
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # Save the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logging.info(f"Successfully downloaded: {filepath}")
            
            # Verify file size
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Less than 1KB
                os.remove(filepath)
                logging.warning(f"Removed invalid PDF file (too small): {filepath}")
                return None

            return filepath

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading PDF from {url}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error saving PDF from {url}: {str(e)}")
            return None

    def generate_search_queries(self, company_name):
        """
        Generate focused search queries for finding official sustainability reports
        """
        base_queries = [
            f"{company_name} sustainability report filetype:pdf",
            f"{company_name} environmental report filetype:pdf",
            f"{company_name} esg report filetype:pdf",
            f"{company_name} corporate responsibility report filetype:pdf"
        ]
        
        # Add year-specific queries for recent years
        current_year = datetime.now().year
        for year in range(current_year - 2, current_year + 1):
            base_queries.extend([
                f"{company_name} sustainability report {year} filetype:pdf",
                f"{company_name} environmental report {year} filetype:pdf",
                f"{company_name} esg report {year} filetype:pdf"
            ])
        
        return base_queries

    def get_company_domain(self, company_name):
        """
        Find company's official domain through web search
        """
        try:
            results = []
            for url in search(f"{company_name} official website", num_results=3):
                results.append(url)
            
            for result in results:
                domain = urlparse(result).netloc
                # Basic validation of company domain
                company_words = company_name.lower().split()
                if any(word in domain.lower() for word in company_words):
                    return domain
            
            return None
                
        except Exception as e:
            logging.error(f"Error finding domain for {company_name}: {str(e)}")
            return None

    def search_sustainability_reports(self, company_name):
        """
        Search for sustainability reports using web search
        """
        potential_urls = set()
        queries = self.generate_search_queries(company_name)
        
        for query in queries:
            try:
                logging.info(f"Searching with query: {query}")
                search_results = search(
                    query,
                    num_results=5,
                    lang="en"
                )
                
                for url in search_results:
                    if self.verify_url(url, company_name):
                        potential_urls.add(url)
                        logging.info(f"Found potential report URL: {url}")
                
                time.sleep(2)  # Be nice to search engines
                
            except Exception as e:
                logging.error(f"Error searching for query '{query}': {str(e)}")
                continue
        
        return list(potential_urls)

    def verify_url(self, url, company_name):
        """
        Verify if URL is likely to be an official sustainability report
        """
        try:
            url_lower = url.lower()
            
            # First, verify it's a PDF
            if not url_lower.endswith('.pdf'):
                return False
            
            # Get the filename from URL
            filename = url_lower.split('/')[-1]
            
            # Check for recent years (2020-2024)
            years = [str(year) for year in range(2020, 2025)]
            has_recent_year = any(year in filename or year in url_lower for year in years)
            
            # Check for report indicators in filename
            report_indicators = [
                'sustainability',
                'environmental',
                'esg',
                'annual',
                'report',
                'progress'
            ]
            
            is_report = any(indicator in filename for indicator in report_indicators)
            
            # Check for exclusion terms
            exclusion_terms = [
                'presentation',
                'brochure',
                'factsheet',
                'fact-sheet',
                'summary',
                'highlights',
                'marketing',
                'press',
                'release'
            ]
            
            has_exclusion = any(term in filename for term in exclusion_terms)
            
            return has_recent_year and is_report and not has_exclusion
            
        except Exception as e:
            logging.error(f"Error verifying URL {url}: {str(e)}")
            return False
        
    def verify_company_domain(self, company_name):
        """
        Get and verify the official company domain
        Returns a list of verified company domains
        """
        if company_name in self.company_domains:
            return self.company_domains[company_name]

        verified_domains = set()
        common_corporate_domains = []

        # Handle special cases
        company_mapping = {
            "Apple Inc": ["apple.com"],
            "Microsoft Corporation": ["microsoft.com", "azure.com"],
            "Google": ["google.com", "alphabet.com"],
            "Amazon": ["amazon.com", "aws.amazon.com"],
            # Add more companies as needed
        }

        if company_name in company_mapping:
            verified_domains.update(company_mapping[company_name])
        else:
            # Search for company website
            try:
                search_query = f"{company_name} official corporate website"
                results = list(search(search_query, num_results=5))
                
                for result in results:
                    domain = urlparse(result).netloc.lower()
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    
                    # Basic company name verification in domain
                    company_words = company_name.lower().split()
                    main_company_word = company_words[0]  # Usually first word is company name
                    
                    # Check if main company word is in domain
                    if main_company_word in domain:
                        verified_domains.add(domain)
                        
            except Exception as e:
                logging.error(f"Error finding domain for {company_name}: {str(e)}")

        self.company_domains[company_name] = list(verified_domains)
        return list(verified_domains)

    def is_valid_company_url(self, url, company_name):
        """
        Check if URL belongs to the company's official domain
        """
        try:
            url_domain = urlparse(url).netloc.lower()
            if url_domain.startswith('www.'):
                url_domain = url_domain[4:]
            
            verified_domains = self.verify_company_domain(company_name)
            
            # Check if URL domain matches any verified company domain
            return any(url_domain.endswith(domain) for domain in verified_domains)
            
        except Exception as e:
            logging.error(f"Error validating URL {url}: {str(e)}")
            return False

    def analyze_url_content(self, url, company_name):
        """
        Analyze URL content to determine if it's a sustainability report
        """
        try:
            headers = {'User-Agent': self.ua.random}
            response = requests.get(url, headers=headers, timeout=10)
            
            if url.lower().endswith('.pdf'):
                return self.analyze_pdf_metadata(response, url, company_name)
            else:
                return self.analyze_webpage_content(response, url, company_name)
                
        except Exception as e:
            logging.error(f"Error analyzing content for {url}: {str(e)}")
            return 0
        
    def analyze_pdf_metadata(self, response, url, company_name):
        """
        Analyze PDF metadata to verify it's an official sustainability report
        """
        score = 0
        
        # Check file size (sustainability reports are usually substantial)
        file_size = len(response.content)
        if file_size > 2000000:  # More than 2MB
            score += 30
        elif file_size > 1000000:  # More than 1MB
            score += 20
        
        # Check URL and filename patterns
        url_lower = url.lower()
        filename = url_lower.split('/')[-1]
        
        # Check for recent years
        current_year = datetime.now().year
        years = [str(year) for year in range(current_year - 2, current_year + 1)]
        if any(year in filename for year in years):
            score += 30
        
        # Check for official report indicators
        if 'sustainability-report' in filename or 'environmental-report' in filename:
            score += 40
        elif 'sustainability' in filename and 'report' in filename:
            score += 30
        
        return min(score, 100)

    def analyze_webpage_content(self, response, url, company_name):
        """
        Analyze webpage content to determine if it's an official sustainability report
        """
        score = 0
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check page title
        title = soup.title.string.lower() if soup.title else ''
        if any(term in title for term in ['sustainability report', 'esg report', 'environmental report']):
            score += 30
        
        # Look for specific headers or section titles
        headers = [h.text.lower() for h in soup.find_all(['h1', 'h2', 'h3'])]
        report_headers = [
            'sustainability report',
            'esg report',
            'environmental report',
            'about this report',
            'report overview',
            'sustainability performance',
            'environmental performance'
        ]
        
        if any(any(header in h for header in report_headers) for h in headers):
            score += 30
        
        # Check for report navigation elements
        nav_elements = soup.find_all(['nav', 'menu'])
        for nav in nav_elements:
            nav_text = nav.text.lower()
            if any(term in nav_text for term in ['report contents', 'report navigation', 'chapter']):
                score += 20
                break
        
        # Check for download links to PDF reports
        pdf_links = soup.find_all('a', href=lambda x: x and x.lower().endswith('.pdf'))
        for link in pdf_links:
            link_text = link.text.lower()
            if any(term in link_text for term in ['download report', 'full report', 'pdf version']):
                score += 20
                break
        
        return min(score, 100)

def main():
    finder = SustainabilityReportFinder()
    companies = [
        "Apple Inc",
        # "Microsoft Corporation",
        # "Google",
        # "Amazon",
        # "Meta"
    ]
    
    for company in companies:
        logging.info(f"\nProcessing company: {company}")
        urls = finder.search_sustainability_reports(company)
        
        if not urls:
            logging.info(f"No sustainability reports found for {company}")
            continue
            
        logging.info(f"Found {len(urls)} potential sustainability reports for {company}")
        
        # Analyze, score, and download each URL
        scored_urls = []
        for url in urls:
            try:
                headers = {'User-Agent': finder.ua.random}
                response = requests.get(url, headers=headers, timeout=10)
                score = finder.analyze_pdf_metadata(response, url, company)
                
                # Download if score is above threshold
                if score >= 50:  # Adjust threshold as needed
                    downloaded_path = finder.download_pdf(url, company)
                    scored_urls.append({
                        'url': url,
                        'score': score,
                        'local_path': downloaded_path
                    })
                
            except Exception as e:
                logging.error(f"Error analyzing {url}: {str(e)}")
        
        # Sort by score and display results
        scored_urls.sort(key=lambda x: x['score'], reverse=True)
        
        logging.info(f"\nTop sustainability reports for {company}:")
        for result in scored_urls[:3]:  # Show top 3 results
            logging.info(f"URL: {result['url']}")
            logging.info(f"Confidence Score: {result['score']}")
            logging.info(f"Local File: {result['local_path']}")
            logging.info("---")

if __name__ == "__main__":
    main()