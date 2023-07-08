def dedup_stream(stream):
    uniq = {}
    for ex in stream:
        uniq[hash(ex["text"])] = ex
    for ex in uniq.values():
        yield ex

def add_rownum(stream):
    for i, ex in enumerate(stream):
        yield {"text": ex["text"], "idx": i}