import time 
import srsly 
import h5py
import modal


stub = modal.Stub("example-get-started")
image = (modal.Image.debian_slim()
         .pip_install("simsity", "embetter[text]", "h5py")
         .run_commands("python -c 'from embetter.text import SentenceEncoder; SentenceEncoder()'"))


@stub.function(image=image, gpu="any")
def create(data):
    from embetter.text import SentenceEncoder

    return SentenceEncoder().transform(data)


@stub.local_entrypoint()
def main():
    tic = time.time()
    data = list(srsly.read_jsonl("data/sentences.jsonl"))[:100_000]
    X = create.call(data)
    with h5py.File('embeddings.h5', 'w') as hf:
        hf.create_dataset("embeddings",  data=X)
    toc = time.time() 
    print(f"took {toc - tic}s to embed shape {X.shape}")
