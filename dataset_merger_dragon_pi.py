import pandas as pd
import os

# Set the directory to where your script and folders live
work_dir = r"C:\Users\Naren\PycharmProjects\PythonProject\.venv"
os.chdir(work_dir)

print("[+] Loading normal baseline 1...")
df_norm1 = pd.read_csv(r".venv/pi_norm_large/pi_norm_1.csv")
df_norm1['session'] = 1  # Unique ID for this capture

print("[+] Loading normal baseline 2...")
df_norm2 = pd.read_csv(r".venv/pi_norm_large/pi_norm_2.csv")
df_norm2['session'] = 2

print("[+] Loading DoS attack data (SYN Flood)...")
df_dos = pd.read_csv(r".venv/pi_dos_large/pi_synflood.csv")
df_dos['session'] = 3

print("[+] Merging into a single dataset...")
final_df = pd.concat([df_norm1, df_norm2, df_dos], ignore_index=True)

print("[+] Saving dragon_pi_dataset.csv (This may take a minute)...")
final_df.to_csv("dragon_pi_dataset.csv", index=False)

print("[+] Merge Complete! You can now run your Edge IDS V4 training script.")