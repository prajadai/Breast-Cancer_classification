import zipfile

# Extract to current directory
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
