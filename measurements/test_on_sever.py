import requests
import time 

start_time = time.time() 

# URL of the API endpoint
url = "http://3.15.239.14:8000/full_prediction/"

# Path to the file to be uploaded
file_path = "/home/xinshuo3/FashionSys/measurements/samples/shoes.jpg"

# Prepare the 'files' dictionary to send with the POST request.
# The key 'file' corresponds to the name of the form field for file upload.
# The tuple consists of the filename, the open file object, and the file's MIME type (optional).
files = {
    'file': ('shoes.jpg', open(file_path, 'rb'), 'image/jpeg')
}

# Make the POST request with the file
response = requests.post(url, files=files)

# Close the file after the request is made to release resources
files['file'][1].close()

# Check the response
if response.status_code == 200:
    end_time = time.time() 
    duration = end_time - start_time 
    print(f"Time taken for calling api: {duration} seconds")
else:
    print("Error:", response.status_code, response.text)
