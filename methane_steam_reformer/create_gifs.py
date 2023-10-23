"""
Created on Tue Apr 13 2023

@author: Alexander Keßler
"""

import cv2
import os
import re

# Input Parameter
path = "/home/alexander/Dokumente/F2_Praktikum_Python/methane_steam_reformer/plots"
prefix = "temperature" #causal_weights,mole_fraction,temperature

def extract_number(filename):
    # Extract the number at the end of the file name
    match = re.search(r'(\d+)\.png$', filename)
    return int(match.group(1)) if match else -1

def find_and_sort_files(directory, prefix):
    # Searches for files with the given prefix in the directory
    files = [file for file in os.listdir(directory) if file.startswith(prefix) and file.endswith(".png")]

    # Sort the files according to the extracted number
    sorted_files = sorted(files, key=extract_number)

    return sorted_files

def create_video(file_paths, output_path, fps=24):
    # Check whether the file paths are present
    if not file_paths:
        raise ValueError("Die Liste der Dateipfade ist leer.")

    # Read single image to get the dimensions
    sample_image = cv2.imread(file_paths[0])
    height, width, _ = sample_image.shape

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec: mp4v is MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        # Add images to the video
        for file_path in file_paths:
            frame = cv2.imread(file_path)
            out.write(frame)
    except Exception as e:
        print(f"Fehler beim Hinzufügen von Bildern zum Video: {e}")
    finally:
        # Close VideoWriter
        out.release()

if __name__ == "__main__":
    # Find and sort the files
    file_names = find_and_sort_files(path, prefix)
    
    # Create file paths of images
    image_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(path, file_name))

    # Create movie    
    movie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),prefix+".mp4")
    create_video(image_paths,movie_path,fps=5)
