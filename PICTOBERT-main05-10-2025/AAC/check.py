import requests
url = "https://drive.google.com/uc?export=view&id=1pvQndd86VYsPxk7yEgQNH3FG3epx-kfd"
r = requests.get(url)
print(r.status_code, len(r.content))
