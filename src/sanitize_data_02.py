import os
def sanitize_filename(filename):
    """
    Sanitize the filename by replacing spaces with underscores and making the file extension lowercase.

    Args:
    filename (str): The name of the file to sanitize.

    Returns:
    str: The sanitized filename.
    """
    name, ext = os.path.splitext(filename)
    sanitized_name = name.replace(' ', '_')  # Replace spaces with underscores
    return sanitized_name + ext.lower()  # Ensure the extension is in lowercase
def sanitize_directory(directory):
    """
    Sanitize all file names in the directory.

    Args:
    directory (str): The directory path where files need to be sanitized.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            sanitized_name = sanitize_filename(file)
            original_file = os.path.join(root, file)
            sanitized_file = os.path.join(root, sanitized_name)

            if original_file != sanitized_file:
                os.rename(original_file, sanitized_file)
                print(f'Renamed: {original_file} -> {sanitized_file}')


if __name__ == '__main__':
    directory_path = '../raw_data'  # Update this to the directory you want to sanitize
    sanitize_directory(directory_path)
