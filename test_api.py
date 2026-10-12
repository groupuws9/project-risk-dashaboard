import json
import requests

url = "http://localhost:5000/api/insights"
response = requests.get(url, timeout=5)
print("Status:", response.status_code)
print("Content-Type:", response.headers.get("Content-Type"))
print("Response body:", response.text[:1000])
