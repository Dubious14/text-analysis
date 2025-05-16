import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import os


def lowercase_file():
    """
    Opens a file dialog to select a text file, converts words in the second column to lowercase,
    and rewrites the file in-place while displaying a progress bar.
    This version processes the file line by line to reduce RAM usage.
    """
    # Open file selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide root window

    input_file = filedialog.askopenfilename(title="Select File to Modify",
                                            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if not input_file:
        print("No file selected. Exiting...")
        return

    temp_file = input_file + ".tmp"  # Temporary file for safe writing

    # Count total lines for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Process file line by line and write to temp file
    with open(input_file, "r", encoding="utf-8") as infile, \
            open(temp_file, "w", encoding="utf-8") as outfile, \
            tqdm(total=total_lines, desc="Processing", unit="lines") as pbar:

        for line in infile:
            line = line.strip()
            if line:
                # Handle both tab-separated and space-separated formats
                parts = line.split("\t") if "\t" in line else line.split()

                if len(parts) > 1:  # Ensure it has at least two columns
                    parts[1] = parts[1].lower()  # Lowercase the second column

                outfile.write("\t".join(parts) + "\n")  # Preserve tab structure if it exists
            else:
                outfile.write("\n")  # Preserve sentence breaks

            pbar.update(1)  # Update progress bar

    # Replace original file with modified version
    os.replace(temp_file, input_file)

    print(f"File successfully updated: {input_file}")


# Run the script
lowercase_file()
