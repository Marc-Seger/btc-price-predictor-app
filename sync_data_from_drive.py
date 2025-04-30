import gdown
import os

# Define your target folder (adjust if your path is different)
target_folder = '/Users/marcseger/Code/GitHubMarc/bitcoin-market-dashboard/data'

# List of files to download: {filename: file_id}
files_to_download = {
    "master_df_dashboard.csv": "1OB7KLiqqolyNDYTIyMlEUfD_D4w1Otii",
    "ETF_Flow_Cleaned.csv": "1nrPOstk9RilKetaU3X-HVyYEo2U10PtS",
    "multiTimeline.csv": "1kF2dmRfrTFG4suOGh825ouCtseJ6EA7y"
}

# Download each file
for filename, file_id in files_to_download.items():
    url = f'https://drive.google.com/uc?id={file_id}'
    output_path = os.path.join(target_folder, filename)
    print(f"⬇️ Downloading {filename} ...")
    gdown.download(url, output_path, quiet=False)

print("\n✅ All files downloaded successfully into your local data folder!")
