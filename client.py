import requests
import json

# File path of the video to be sent for detection
video_path = "highway-4.mp4"

# Read video data
with open(video_path, 'rb') as file:
    video_data = file.read()

# Compose payload with video data
files = {'video': ('highway-4.mp4', video_data)}
# print(files)

# Sending a post request to the server (API) for object detection
response = requests.post(url="http://192.168.68.122:5000/detect", files=files)

# Printing out the response of API
print(response.json())
