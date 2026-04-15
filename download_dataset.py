#!/usr/bin/env python3
"""Download and prepare a 1-3 GB subset of Stack Exchange data."""

import os
import sys
import requests
import subprocess
import shutil
from tqdm import tqdm

def download_file(url, dest):
    """Download with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=dest) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def main():
    # Ask Ubuntu dump (small site, ~2.5 GB compressed)
    url = "https://archive.org/download/stackexchange/math.stackexchange.com.7z"
    archive_path = "math.stackexchange.com.7z"
    extract_dir = "data/raw"
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print("Downloading Ask Ubuntu dump (2.5 GB)...")
    if not os.path.exists(archive_path):
        download_file(url, archive_path)
    else:
        print("Archive already exists.")
    
    print("Extracting 7z archive (requires p7zip)...")
    subprocess.run(["7z", "x", archive_path, f"-o{extract_dir}", "-y"], check=True)
    # prefer system 7z if available, otherwise use py7zr
    if shutil.which("7z"):
        subprocess.run(["7z", "x", archive_path, f"-o{extract_dir}", "-y"], check=True)
    else:
        try:
            import py7zr
        except ImportError:
            print("py7zr not installed. Install with: pip install py7zr")
            sys.exit(1)
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=extract_dir)

    # The extracted file is Posts.xml (multiple GB). We'll take only first 2M rows.
    posts_xml = os.path.join(extract_dir, "Posts.xml")
    subset_xml = os.path.join(extract_dir, "Posts_subset.xml")
    print("Creating 2 million row subset (approx 1.5 GB)...")
    with open(posts_xml, "r") as infile, open(subset_xml, "w") as outfile:
        outfile.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<posts>\n")
        row_count = 0
        for line in infile:
            outfile.write(line)
            if line.strip().startswith("<row"):
                row_count += 1
                if row_count >= 2_000_000:
                    break
        outfile.write("</posts>\n")
    print(f"Subset created: {subset_xml} (size: {os.path.getsize(subset_xml) / 1e9:.2f} GB)")
    print("Done. Dataset ready at", subset_xml)

if __name__ == "__main__":
    main()


# import py7zr
# py7zr.SevenZipFile('askubuntu.7z', mode='r').extractall(path='data1/raw')
# print('extracted to data/raw')
