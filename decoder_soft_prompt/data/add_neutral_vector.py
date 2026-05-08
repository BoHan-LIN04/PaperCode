import numpy as np

vecs = np.load("decoder_soft_prompt/data/emotion_vectors_orth.npy")
mean_vec = np.mean(vecs, axis=0, keepdims=True)
vecs_new = np.vstack([vecs, mean_vec])
np.save("decoder_soft_prompt/data/emotion_vectors_orth.npy", vecs_new)
print("补齐后shape:", vecs_new.shape)
