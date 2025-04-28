from FAISSrun import FAISSManager # Your builder class
from PIL import Image
import argparse
#Data structure
#Data/
# │  ──object1/
# │      ├── object1.obj
# │      ├── texture.png
# │      ├── .mtl
# │      ├── Views of 3D object1 in 2D/
# │      │     ├── img1.jpg
# │      │     ├── img2.jpg
# │      │── text_description_for_obj1.txt
# │  ──object2/
# │     ├── object2.obj
# │     ├── texture.png
# │     ├── .mtl
# │     ├── views of 3D object2 in 2D/
# │     │     ├── img1.jpg
# │     │     ├── img2.jpg
# │     │──text_description_for_obj2.txt 
# │ 
# │  ──object1/
# │      ├── object1.obj
# │      ├── texture.png
# │      ├── .mtl
# │      ├── Views of 3D object1 in 2D/
# │      │     ├── img1.jpg
# │      │     ├── img2.jpg
# │      │── text_description_for_obj1.txt
# │  ──object2/
# │     ├── object2.obj
# │     ├── texture.png
# │     ├── .mtl
# │     ├── views of 3D object2 in 2D/
# │     │     ├── img1.jpg
# │     │     ├── img2.jpg
# │     │──text_description_for_obj2.txt 
# Step 1: Build database from your dataset
parser = argparse.ArgumentParser(description='Run pipeline')
parser.add_argument('--object_dir', required=True, help='Input dir')
arg = parser.parse_args()
faiss_manager = FAISSManager(embedding_dim=1024)
object_root = arg.object_dir
faiss_manager.build(object_root)  
faiss_manager.save("/root/faiss_index")  # This saves indexes + metadata + object root
faiss_manager.load(save_dir="/root/faiss_index")

# # Perform search with both query and answer embeddings
results,answer_embeddings = faiss_manager.search(
    query="a wooden dresser featuring drawers with yellow circular knobs, showcasing a light-colored wood grain finish.",
    query_type="text",
    search_in="text", 
    top_k=5, # search_in = 'image' or 'text' return top_k*5 answer, search_in = 'mean pooling images' return top_k
    return_answer_vector=True
)

# print("Answer Embedding Shape for the First Result:", answer_embeddings[0].shape)

# Print the results
for r, ans_embed in zip(results, answer_embeddings):
    print(f"Found: {r['object_dir']} with confidence {r['confidence']:.3f}")
    print(f"Answer Vector Shape: {ans_embed.shape}")
