import requests
import json
from datetime import datetime

url = "http://127.0.0.1:8000/predict"

# Test Case 1: Normal Morning
payload1 = {
    "date": "2026-04-15",
    "shift": "Morning",
    "appointments_booked": 120
}

# Test Case 2: Sunday (Should fail)
payload2 = {
    "date": "2026-04-12",
    "shift": "Morning",
    "appointments_booked": 0
}

def test_api(payload, name):
    print(f"--- Testing {name} ---")
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    test_api(payload1, "Case 1: Normal Morning")
    test_api(payload2, "Case 2: Sunday (Expected Error)")
