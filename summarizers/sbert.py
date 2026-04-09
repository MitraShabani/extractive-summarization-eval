from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

"""
I'm using SBERT (Sentence-BERT) which is a general-purpose semantic model.
What it has learned during training = how to assign similar meanings(addresses) in the same neighborhood.
"""
model  = SentenceTransformer("all-MiniLM-L6-v2")

def summarize_sbert(data, compression_ratio=0.3):

    sentences = []
    metadata = []

    final_summary = []

    for block in data:
        sentences.append(block["sentence"])
        metadata.append({
            "header": block["header"],
            "page": block["page"]
        })

    def get_top_k_sentences(sentences, metadata):

        """
        convert raw text (sentences or paragraphs) into numerical vectors.
        In fact, rows = sentences, columns = learned dimensions during training, values = coordinates for each sentence in meaning-space(columns)
        """
        embeddings = model.encode(sentences)

        """
        'cosine_similarity' does the comparison and compares two sentences coordinates.
        In fact, two addresses that are close on the map = similar meaning
        cosine_similarity is just measuring that distance.
        embeddings = vectors for all sentences
        doc_embedding = one vector that representing the average of all vectors
        """
        doc_embedding = embeddings.mean(axis=0, keepdims=True)
        scores = cosine_similarity(embeddings, doc_embedding).flatten() # flatten() -> collapses multi-dimensional arrays into a single-dimensional array

        top_k = max(1, int(len(sentences) * compression_ratio))
        top_indices = set(scores.argsort()[::-1][:top_k])  # sort the array in descending order

        for i in sorted(top_indices):
            final_summary.append({
                "header": metadata[i]["header"],
                "sentence": sentences[i],
                "page": metadata[i]["page"]
            })
        return final_summary

    final_summary = get_top_k_sentences(sentences, metadata)

    return final_summary