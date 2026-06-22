import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="exam_prep")

all_data = collection.get(limit=1000)
from collections import Counter
subjects = Counter(m.get("subject", "MISSING") if m else "NONE_META" for m in all_data["metadatas"])
print(subjects)
print("Total chunks:", len(all_data["ids"]))