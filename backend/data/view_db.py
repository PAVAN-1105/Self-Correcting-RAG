import chromadb
from pprint import pprint

# 1. Connect to the local database directory
client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. Get the specific collection (table) we created
collection = client.get_collection(name="scrag_knowledge_base")

# 3. Print out some basic statistics
print(f"Total chunks in database: {collection.count()}")

# 4. Fetch the first 3 items in the database to inspect them
results = collection.peek(limit=3)

print("\n--- Sample Data ---")
for i in range(len(results['ids'])):
    print(f"\nID: {results['ids'][i]}")
    print(f"Source metadata: {results['metadatas'][i]}")
    print(f"Text content: {results['documents'][i][:150]}...") # Truncating for readability
