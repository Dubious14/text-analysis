import numpy as np
from clean_and_split import prepare_text_from_string
from global_model import get_vector, load_model

# === Load the model ===
load_model("C:/Users/alons/OneDrive/שולחן העבודה/Text_Complexity/Models/word2vecf.npz")

# === Replace this with any sample text ===
sample_text = """
When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.
The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.
At each end of the room, on the wall, hung a beautiful bear-skin rug.
These rugs were for prizes, one for the girls and one for the boys. And this was the game.
The girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.
This would have been an easy matter, but each traveller was obliged to wear snowshoes.
"""

# === Step 1: Clean and tokenize the text ===
sentences = prepare_text_from_string(sample_text)
tokens = [word for sent in sentences for word in sent]

# === Step 2: Retrieve embeddings ===
embeddings = [vec for w in tokens if (vec := get_vector(w)) is not None]


# === Step 3: Compute determinant of the covariance matrix ===
if len(embeddings) >= 2:
    embeddings = np.array(embeddings)
    cov = np.cov(embeddings.T)
    print("Number of tokens with embeddings:", len(embeddings))
    print("Embedding shape:", embeddings.shape)
    print("Rank of covariance matrix:", np.linalg.matrix_rank(cov))
    print("Smallest eigenvalue:", np.min(np.linalg.eigvals(cov)))
    det_cov = np.linalg.det(cov)
    print(f"✅ Determinant of covariance matrix (det_cov): {det_cov:.4e}")
else:
    print("⚠️ Not enough embeddings to compute covariance matrix.")
