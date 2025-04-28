import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoModel

class FAISSManager:
    def __init__(self, embedding_dim=1024, device=None, index_dir=None, model_path="jinaai/jina-clip-v2"):
        """
        Initialize the FAISS Manager for both building and retrieving from FAISS indexes
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            device: Device to use for model inference ('cuda' or 'cpu')
            index_dir: Directory containing existing indexes to load (optional)
        """
        print("[DEBUG] Initializing FAISSManager")
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embedding_dim = embedding_dim
        
        # Load CLIP model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
        self.model.to(self.device)
        
        # Initialize FAISS indexes
        # resource = faiss.StandardGpuResources()
        # image_index_cpu = faiss.IndexFlatIP(self.embedding_dim)
        # text_index_cpu = faiss.IndexFlatIP(self.embedding_dim)
        # mean_pooling_cpu = faiss.IndexFlatIP(self.embedding_dim)
        # self.image_index = faiss.index_cpu_to_gpu(resource, device = self.device,  index =image_index_cpu)
        # self.text_index = faiss.index_cpu_to_gpu(resource, device = self.device, index = text_index_cpu)
        # self.mean_pooling_image_index = faiss.index_cpu_to_gpu(resource,device = self.device,  index = mean_pooling_cpu )
        try:
            self.res = faiss.StandardGpuResources()
            self.image_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            self.text_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            self.mean_pooling_image_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
        except:
            self.image_index = faiss.IndexFlatIP(self.embedding_dim)
            self.text_index = faiss.IndexFlatIP(self.embedding_dim)
            self.mean_pooling_image_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Metadata mapping
        self.image_object_dirs = []
        self.text_object_dirs = []
        self.mean_pooling_image_dirs = []
        self.image_object_root = []
        self.text_object_root = []
        self.mean_pooling_image_root = []
        # Load existing indexes if provided
        if index_dir and os.path.exists(index_dir):
            self.load(index_dir)
    
    def encode_image(self, image_path):
        """Encode a single image"""
        emb = self.model.encode_image([image_path], truncate_dim=self.embedding_dim)[0]#, truncate_dim=self.embedding_dim)[0]
        return emb / np.linalg.norm(emb)
    
    def encode_image_batch(self, image_paths, batch_size=8):
        """Encode a batch of images"""
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            emb = self.model.encode_image(batch, truncate_dim=self.embedding_dim)#, truncate_dim=self.embedding_dim)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)
    
    def encode_text(self, text):
        """Encode text"""
        if isinstance(text, str):
            text = [text]
        emb = self.model.encode_text(text, truncate_dim=self.embedding_dim)#, truncate_dim=self.embedding_dim)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0] if len(text) == 1 else emb
    
    def build(self, data_root, image_batch_size=8):
        """
        Build FAISS indexes from dataset directory
        
        Args:
            data_root: Root directory containing object data
            image_batch_size: Batch size for image encoding
        """
        object_dirs = [os.path.join(data_root, object) for object in os.listdir(data_root) 
                     if os.path.isdir(os.path.join(data_root, object))]
        print('object dirs:', object_dirs[:5])
        
        # object_dirs = []
        # for room in room_dirs:
        #     object_dirs += [os.path.join(room, obj) for obj in os.listdir(room) 
        #                    if os.path.isdir(os.path.join(room, obj))]
        
        # print('Object full dirs:', object_dirs[:5])
        
        for obj_dir in tqdm(object_dirs, desc="Processing objects"):
            obj_name = os.path.basename(obj_dir)
            print(f"[INFO] Processing {obj_name}")
            
            view_subdirs = [d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))]
            demo_img = [f for f in os.listdir(obj_dir) if f.lower().endswith('.jpg')]
            if not demo_img:
                continue
            demo_img_embeds = self.encode_image(os.path.join(obj_dir, demo_img[0]))
            
            if not view_subdirs:
                print(f"[WARNING] No 2D view directory in {obj_name}")
                continue
            
            view_dir = os.path.join(obj_dir, view_subdirs[0])
            img_files = [os.path.join(view_dir, f) for f in os.listdir(view_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if img_files:
                img_embeds = self.encode_image_batch(img_files, batch_size=image_batch_size)
                img_embeds = np.vstack([img_embeds, demo_img_embeds])
                
                for embed in img_embeds:
                    self.image_index.add(np.expand_dims(embed.astype(np.float32), axis=0))
                    self.image_object_dirs.append(obj_name)
                    # self.image_object_root.append(obj_dir)
                
                mean_img_embed = np.mean(img_embeds, axis=0)
                mean_img_embed = mean_img_embed / np.linalg.norm(mean_img_embed)
                self.mean_pooling_image_index.add(np.expand_dims(mean_img_embed.astype(np.float32), axis=0))
                self.mean_pooling_image_dirs.append(obj_name)
                # self.mean_pooling_image_root.append(obj_dir)
            
            text_files = [f for f in os.listdir(obj_dir) if f.lower().endswith(".txt")]
            if text_files:
                text_path = os.path.join(obj_dir, text_files[0])
                try:
                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            text_embed = self.encode_text(text)
                            self.text_index.add(np.expand_dims(text_embed.astype(np.float32), axis=0))
                            self.text_object_dirs.append(obj_name)
                            # self.text_object_root.append(obj_dir)
                except Exception as e:
                    print(f"[WARNING] Failed to read text for {obj_name}: {e}")
        
        print("[INFO] Database build finished.")
    
    def add(self, embeddings, object_dirs, target="image"):
        """
        Add embeddings to specified index
        
        Args:
            embeddings: np.array of shape (N, D)
            object_dirs: list of metadata strings (same length as embeddings)
            target: 'image', 'text', or 'mean pooling images'
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        
        if target == "image":
            self.image_index.add(embeddings)
            # self.image_object_dirs.extend(object_dirs)
        elif target == "text":
            self.text_index.add(embeddings)
            # self.text_object_dirs.extend(object_dirs)
        elif target == "mean pooling images":
            self.mean_pooling_image_index.add(embeddings)
            # self.mean_pooling_image_dirs.extend(object_dirs)
        else:
            raise ValueError("target must be 'image', 'text', or 'mean pooling images'")
    
    def search(self, query, query_type="text", top_k=5, search_in="image", return_answer_vector=False):
        """
        Search for similar objects
        
        Args:
            query: str (text) or str (path to image)
            query_type: 'text' or 'image'
            top_k: Number of results to return
            search_in: 'image', 'text', or 'mean pooling images'
            return_answer_vector: If True, returns (results, query_embedding, answer_embedding)
            
        Returns:
            List of dictionaries with object_dir and confidence, and optionally embeddings
        """
        if query_type == "text":
            query_vector = self.encode_text(query)
        elif query_type == "image":
            query_vector = self.encode_image(query)
        else:
            raise ValueError("query_type must be 'text' or 'image'")
        
        query_vector = query_vector[np.newaxis, :]
        
        if search_in == "image":
            index = self.image_index
            object_dirs = self.image_object_dirs
            # object_root = self.image_object_root
            limit = top_k * 7
        elif search_in == "text":
            index = self.text_index
            object_dirs = self.text_object_dirs
            # object_root = self.text_object_root
            limit = top_k * 7
        elif search_in == "mean pooling images":
            index = self.mean_pooling_image_index
            object_dirs = self.mean_pooling_image_dirs
            # object_root = self.mean_pooling_image_root
            limit = top_k
        else:
            raise ValueError("search_in must be 'image', 'text', or 'mean pooling images'")
        
        distances, indices= index.search(query_vector, limit)
        
        hits = []
        answer_embeddings = []  # To store answer embeddings
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            hits.append({
                "object_dir": object_dirs[idx],
                "confidence": dist,
                # "object_root": object_root[idx]
            })
            # Store the answer embeddings
            answer_embeddings.append(index.reconstruct(int(idx)))  # Reconstruct the embedding from the index
        
        # Deduplicate if not mean-pooling
        output = []
        if search_in == "mean pooling images":
            output = hits[:top_k]
        else:
            hit_set = set()
            for hit in hits:
                if hit["object_dir"] not in hit_set:
                    hit_set.add(hit["object_dir"])
                    output.append(hit)
                    if len(output) == top_k:
                        break
        
        if return_answer_vector:
            return output, answer_embeddings
        else:
            return output
    
    def batch_search(self, queries, query_type="text", top_k=5, search_in="image"):
        """
        Batch search for multiple queries at once
        
        Args:
            queries: List of queries (text strings or image paths)
            query_type: 'text' or 'image'
            top_k: Number of results to return per query
            search_in: 'image', 'text', or 'mean pooling images'
            
        Returns:
            List of lists of dictionaries with object_dir and confidence
        """
        if query_type == "text":
            query_vectors = np.array([self.encode_text(query) for query in queries])
        elif query_type == "image":
            query_vectors = np.array([self.encode_image(query) for query in queries])
        else:
            raise ValueError("query_type must be 'text' or 'image'")
        
        if search_in == "image":
            index = self.image_index
            object_dirs = self.image_object_dirs
        elif search_in == "text":
            index = self.text_index
            object_dirs = self.text_object_dirs
        elif search_in == "mean pooling images":
            index = self.mean_pooling_image_index
            object_dirs = self.mean_pooling_image_dirs
        else:
            raise ValueError("search_in must be 'image', 'text', or 'mean pooling images'")
        
        distances, indices = index.search(query_vectors, top_k)
        
        results = []
        for idx_set, dist_set in zip(indices, distances):
            batch_hits = []
            for idx, dist in zip(idx_set, dist_set):
                if idx == -1:
                    continue
                batch_hits.append({
                    "object_dir": object_dirs[idx],
                    "confidence": dist
                })
            results.append(sorted(batch_hits, key=lambda x: -x["confidence"]))
        
        return results
    
    def save(self, save_dir):
        """
        Save indexes and metadata
        
        Args:
            save_dir: Directory to save indexes and metadata
        """
        os.makedirs(save_dir, exist_ok=True)
        self.image_index = faiss.index_gpu_to_cpu(self.image_index)
        self.text_index = faiss.index_gpu_to_cpu(self.text_index)
        self.mean_pooling_image_index = faiss.index_gpu_to_cpu(self.mean_pooling_image_index)
        faiss.write_index(self.image_index, os.path.join(save_dir, "image_index.faiss"))
        faiss.write_index(self.text_index, os.path.join(save_dir, "text_index.faiss"))
        faiss.write_index(self.mean_pooling_image_index, os.path.join(save_dir, "mean_image_index.faiss"))
        
        np.save(os.path.join(save_dir, "image_object_dirs.npy"), np.array(self.image_object_dirs))
        np.save(os.path.join(save_dir, "text_object_dirs.npy"), np.array(self.text_object_dirs))
        np.save(os.path.join(save_dir, "mean_pooling_image_dirs.npy"), np.array(self.mean_pooling_image_dirs))
        # np.save(os.path.join(save_dir, "image_object_root.npy"), np.array(self.image_object_root))
        # np.save(os.path.join(save_dir, "text_object_root.npy"), np.array(self.text_object_root))
        # np.save(os.path.join(save_dir, "mean_pooling_image_root.npy"), np.array(self.mean_pooling_image_root))       
        print("[INFO] Index and metadata saved.")
    
    def load(self, save_dir):
        """
        Load indexes and metadata
        
        Args:
            save_dir: Directory containing indexes and metadata
        """
        try:
            # Load indexes

            image_index = faiss.read_index(os.path.join(save_dir, "image_index.faiss"))
            text_index = faiss.read_index(os.path.join(save_dir, "text_index.faiss"))
            mean_pooling_image_index = faiss.read_index(os.path.join(save_dir, "mean_image_index.faiss"))

            self.image_index = faiss.index_cpu_to_gpu(self.res, 0, image_index)
            self.text_index = faiss.index_cpu_to_gpu(self.res, 0, text_index)
            self.mean_pooling_image_index = faiss.index_cpu_to_gpu(self.res, 0, mean_pooling_image_index)

            
            # Load metadata
            self.image_object_dirs = np.load(os.path.join(save_dir, "image_object_dirs.npy"), allow_pickle=True).tolist()
            self.text_object_dirs = np.load(os.path.join(save_dir, "text_object_dirs.npy"), allow_pickle=True).tolist()
            self.mean_pooling_image_dirs = np.load(os.path.join(save_dir, "mean_pooling_image_dirs.npy"), allow_pickle=True).tolist()
            # self.image_object_root = np.load(os.path.join(save_dir, "image_object_root.npy"), allow_pickle=True).tolist()
            # self.text_object_root = np.load(os.path.join(save_dir, "text_object_root.npy"), allow_pickle=True).tolist()
            # self.mean_pooling_image_root = np.load(os.path.join(save_dir, "mean_pooling_image_root.npy"), allow_pickle=True).tolist()   
            print("[INFO] Index and metadata loaded successfully.")
        except Exception as e:
            self.image_index = faiss.read_index(os.path.join(save_dir, "image_index.faiss"))
            self.text_index = faiss.read_index(os.path.join(save_dir, "text_index.faiss"))
            self.mean_pooling_image_index = faiss.read_index(os.path.join(save_dir, "mean_image_index.faiss"))
            self.image_object_dirs = np.load(os.path.join(save_dir, "image_object_dirs.npy"), allow_pickle=True).tolist()
            self.text_object_dirs = np.load(os.path.join(save_dir, "text_object_dirs.npy"), allow_pickle=True).tolist()
            self.mean_pooling_image_dirs = np.load(os.path.join(save_dir, "mean_pooling_image_dirs.npy"), allow_pickle=True).tolist()
            # self.image_object_root = np.load(os.path.join(save_dir, "image_object_root.npy"), allow_pickle=True).tolist()
            # self.text_object_root = np.load(os.path.join(save_dir, "text_object_root.npy"), allow_pickle=True).tolist()
            # self.mean_pooling_image_root = np.load(os.path.join(save_dir, "mean_pooling_image_root.npy"), allow_pickle=True).tolist()   
            # print(f"[ERROR] Failed to load indexes: {e}")
