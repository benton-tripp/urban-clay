import re

# Function to extract the band/suffix from the filename
def extract_suffix(filename):
    # Match the pattern after the second timestamp
    match = re.search(r'\d{8}T\d{6}_R\d{3}_[A-Z0-9]+_\d{8}T\d{6}_(.+)\.tif$', filename)
    if match:
        return match.group(1) # This returns the suffix (e.g., 'B02', 'rendered_preview', 'tilejson')
    # If that fails
    match = re.search(r'\d{8}T\d{6}_.+?_(.+)\.tif$', filename)
    if match:
        return match.group(1) 
    return None