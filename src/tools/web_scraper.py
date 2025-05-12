import requests
from bs4 import BeautifulSoup
from typing import Optional
import logging
import time
import os
from urllib.parse import urlparse

# Disable trafilatura's signal handling to prevent issues in threaded environments
os.environ["SIGNAL_HANDLING"] = "0"

# Import trafilatura after setting environment variable
import trafilatura

class WebScraperTool:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }

    def scrape(self, url: str) -> dict:
        try:
            logging.info(f"Starting to scrape URL: {url}")
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "url": url,
                    "error": "Invalid URL format",
                    "status": "failed"
                }
            
            # Download webpage with timeout and retries
            max_retries = 2
            retry_count = 0
            response = None
            
            while retry_count <= max_retries:
                try:
                    response = requests.get(url, headers=self.headers, timeout=15)
                    response.raise_for_status()
                    break
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise e
                    logging.warning(f"Retry {retry_count}/{max_retries} for {url}: {str(e)}")
                    time.sleep(1)  # Short delay before retry
            
            if not response:
                raise Exception("Failed to download page after retries")
            
            # First try BeautifulSoup for extraction (safer in threaded environments)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text from main content areas - try multiple strategies
            main_content = ""
            
            # Strategy 1: Look for common content containers
            main_tags = soup.find_all(['article', 'main', 'div', 'section'], 
                                     class_=lambda c: c and any(x in str(c).lower() for x in 
                                                               ['content', 'main', 'article', 'body', 'post']))
            
            if main_tags:
                main_content = ' '.join([tag.get_text(separator=' ', strip=True) for tag in main_tags])
            
            # Strategy 2: If content is too short, try looking for paragraph tags
            if len(main_content) < 200:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    p_content = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                    if len(p_content) > len(main_content):
                        main_content = p_content
                        logging.info(f"Used paragraph content for {url} ({len(main_content)} chars)")
            
            # Strategy 3: If still too short, get all text
            if len(main_content) < 100:
                body_content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
                if len(body_content) > len(main_content):
                    main_content = body_content
                    logging.info(f"Used body content for {url} ({len(main_content)} chars)")
            
            # Clean up text
            main_content = ' '.join(line.strip() for line in main_content.splitlines() if line.strip())
            
            # Only try trafilatura as a last resort if BeautifulSoup didn't get enough content
            # Completely skip trafilatura in threaded environments to avoid recursion issues
            if len(main_content) < 200:
                try:
                    # Check if we're in the main thread before using trafilatura
                    import threading
                    if threading.current_thread() is threading.main_thread():
                        # Only use trafilatura in the main thread
                        trafilatura_content = trafilatura.extract(response.text, include_comments=False, 
                                                                include_tables=True, favor_precision=True, 
                                                                output_format="text", url=url)
                        
                        if trafilatura_content and len(trafilatura_content) > len(main_content):
                            main_content = trafilatura_content
                            logging.info(f"Used trafilatura content for {url} ({len(main_content)} chars)")
                    else:
                        logging.info(f"Skipping trafilatura for {url} - not in main thread")
                except Exception as e:
                    logging.warning(f"Trafilatura extraction failed for {url}: {str(e)}")
                    # Continue with BeautifulSoup content
            
            # Extract metadata
            metadata = {
                "title": soup.title.string if soup.title else "",
                "meta_description": soup.find("meta", {"name": "description"})["content"] if soup.find("meta", {"name": "description"}) else "",
                "publish_date": soup.find("meta", {"property": "article:published_time"})["content"] if soup.find("meta", {"property": "article:published_time"}) else None
            }
            
            logging.info(f"Successfully scraped {url}, content length: {len(main_content) if main_content else 0} characters")

            return {
                "url": url,
                "content": main_content,
                "metadata": metadata,
                "status": "success"
            }

        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e),
                "status": "failed"
            }



