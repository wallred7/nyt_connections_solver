"""
This module provides functions to fetch and combine JSON data from NYT Connections game.
"""

import requests
import json
from datetime import datetime, timedelta
from time import sleep

def fetch_json(date: str) -> dict:
    """
    Fetches JSON data for a given date from the NYT Connections game API.

    Args:
        date (str): The date in the format YYYY-MM-DD.

    Returns:
        dict: The JSON data for the given date.
    """
    url = f"https://www.nytimes.com/svc/connections/v1/{date}.json"
    response = requests.get(url)
    sleep(0.1)  # To prevent overwhelming the server
    return response.json()

def combine_jsons(start_date: datetime, end_date: datetime) -> list:
    """
    Combines JSON data for a range of dates from the NYT Connections game API.

    Args:
        start_date (datetime): The start date of the range.
        end_date (datetime): The end date of the range.

    Returns:
        list: A list of JSON data for the given date range.
    """
    current_date = start_date
    combined_data = []

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        json_data = fetch_json(date_str)
        combined_data.append(json_data)
        current_date += timedelta(days=1)

    return combined_data

if __name__ == "__main__":
    start_date = datetime(2023, 6, 12)
    end_date = datetime.now()
    combined_data = combine_jsons(start_date, end_date)
    
    with open("src/data/combined.json", "w") as f:
        json.dump(combined_data, f, indent=2)
