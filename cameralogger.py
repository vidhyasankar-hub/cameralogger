# Step 1: Install Pyngrok
!pip install -q pyngrok

# Step 2: Upload your HTML file
from google.colab import files
uploaded = files.upload()
html_filename = list(uploaded.keys())[0]

# Step 3: Start a local HTTP server
import http.server, socketserver
import threading

PORT = 8000

def start_server():
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    print(f"Local server running at http://localhost:{PORT}/")

start_server()

# Step 4: Connect with ngrok
from pyngrok import ngrok
ngrok.set_auth_token("2wdkDkvs65y5zoTJtljuyNtxslk_3Kja2FKZNxjysJjDbVuFK")  # replace with your token
public_url = ngrok.connect(PORT)
print(f"Public URL: {public_url}")

# Step 5: Define a Python callback to collect brightness data
from google.colab import output
import csv

brightness_data = []

def log_brightness(timestamp, brightness):
    brightness_data.append((timestamp, brightness))
    # Save to CSV file
    with open('brightness_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Brightness'])
        writer.writerows(brightness_data)

output.register_callback('logFrameColab', log_brightness)
print("Ready to receive brightness data.")

# Step 6: Provide clickable link to open the HTML
from IPython.display import HTML

print("Click the link below to open your camera logger page:")
HTML(f'<a href="{public_url}/{html_filename}" target="_blank">Open Camera Logger</a>')
