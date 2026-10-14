from remotezip import RemoteZip

# 1. Paste your copied Zenodo Download Link here
url = "https://zenodo.org/records/10784947/files/dragon_pi.zip?download=1"

print("[+] Connecting to Zenodo and reading the giant ZIP index...")

try:
    with RemoteZip(url) as zip_file:
        print("[+] Found the file! Extracting pi_norm_large.zip (675 MB)...")
        # Notice the path matches your screenshot exactly
        zip_file.extract('dragon_pi/pi/pi_norm_large.zip')

        print("[+] Extracting pi_dos_large.zip (103 MB)...")
        zip_file.extract('dragon_pi/pi/pi_dos_large.zip')

    print("[+] Done! Successfully bypassed the 12 GB download.")
except Exception as e:
    print(f"Error: {e}")