#!/usr/bin/env python3
import spacy
import multiprocessing
from tqdm import tqdm

# Load the SpaCy model with dependency parsing enabled
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])  # Keep parser enabled

# Set multiprocessing parameters
num_threads = max(2, multiprocessing.cpu_count() // 2)
batch_size = 5000  # Process 5000 sentences at a time for efficiency
chunksize = batch_size // num_threads  # Ensure efficient distribution of work

def convert_sentence_to_conll(sentence):
    """
    Converts a single sentence into a CoNLL-X formatted string.
    Ensures valid dependency parsing.
    """
    doc = nlp(sentence)
    conll_lines = []

    for token in doc:
        token_id = token.i + 1  # Convert to 1-based index
        form = token.text
        lemma = token.lemma_
        cpos = token.pos_
        pos = token.tag_
        head = token.head.i + 1 if token.head != token else 0  # Assign correct head index
        deprel = token.dep_
        phead, pdeprel = "_", "_"

        # Standard CoNLL line
        line = f"{token_id}\t{form}\t{lemma}\t{cpos}\t{pos}\t{head}\t{deprel}\t{phead}\t{pdeprel}"
        conll_lines.append(line)

    return "\n".join(conll_lines)

def process_sentences(sentences):
    """
    Process sentences in parallel efficiently.
    """
    with multiprocessing.Pool(num_threads) as pool:
        return list(pool.imap_unordered(convert_sentence_to_conll, sentences, chunksize=chunksize))

def main():
    input_file = "../processed_sentences.txt"
    output_file = "../conll_format.txt"

    # Count total sentences for progress bar
    with open(input_file, "r", encoding="utf-8") as infile:
        total_sentences = sum(1 for line in infile if line.strip())

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        sentences = []
        with tqdm(total=total_sentences, desc="Converting to CoNLL", unit="sentences") as progress_bar:
            for line in infile:
                line = line.strip()
                if line:
                    sentences.append(line)

                # Process sentences in optimized batch sizes
                if len(sentences) >= batch_size:
                    for result in process_sentences(sentences):
                        outfile.write(result + "\n\n")
                    sentences.clear()  # Reset batch for next round
                    progress_bar.update(batch_size)

            # Process remaining sentences (final batch)
            if sentences:
                for result in process_sentences(sentences):
                    outfile.write(result + "\n\n")
                progress_bar.update(len(sentences))

    print(f"✅ Conversion complete! CoNLL file saved to '{output_file}'.")

if __name__ == "__main__":
    main()

