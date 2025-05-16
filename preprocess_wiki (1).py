#!/usr/bin/env python3
import re
import nltk
import sys
from tqdm import tqdm

# Download the NLTK tokenizer data if not already installed
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """Cleans and tokenizes Wikipedia-style text into sentences."""
    # Remove custom markers like _START_ARTICLE_, _START_SECTION_, _START_PARAGRAPH_
    text = re.sub(r'_START_[A-Z]+_', '', text)
    # Replace _NEWLINE_ markers with actual newline characters
    text = text.replace('_NEWLINE_', '\n')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize the cleaned text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def main():
    if len(sys.argv) != 3:
        print("Usage: ./script.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Get total lines for progress bar
    with open(input_file, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)
    
    # Process and write output line by line
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing", unit=" line", dynamic_ncols=True):
            sentences = preprocess_text(line)
            outfile.writelines(sentence + "\n" for sentence in sentences)
    
    print(f"✅ Processing complete. Sentences saved to '{output_file}'.")

if __name__ == "__main__":
    main()

