import numpy as np

vecs = np.load("decoder_soft_prompt/data/emotion_vectors_orth.npy")
print("emotion_vectors_orth.npy shape:", vecs.shape)
print("类别数（行数）:", vecs.shape[0])
