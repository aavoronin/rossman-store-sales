import re

def sanitize_filename(filename):
    # Define the characters that are considered invalid in a file name
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'

    # Replace invalid characters with an underscore
    sanitized_filename = re.sub(invalid_chars, '_', filename)

    return sanitized_filename
