import urllib.request
import os

url = "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt"
out_path = os.path.join("models", "edsr_x4.pt")
os.makedirs("models", exist_ok=True)
print(f"Downloading EDSR ×4 weights from {url} ...")
urllib.request.urlretrieve(url, out_path)
print(f"Downloaded EDSR ×4 weights to {out_path}") 