import time 
import h5py
import srsly 
from embetter.text import SentenceEncoder
from simsity import create_index

def create(data):
    return SentenceEncoder().transform(data)

def main():
    tic = time.time()
    data = list(srsly.read_jsonl("data/sentences.jsonl"))[:20000]
    X = create(data)
    with h5py.File('embeddings.h5', 'w') as hf:
        hf.create_dataset("embeddings",  data=X)
    toc = time.time() 
    print(f"took {toc - tic}s to embed shape {X.shape}")

if __name__ == "__main__":
    main()
