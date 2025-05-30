from Database import FAISSManager
from Panotomask import Maskcreation 
from ZeroPainter import Inpainting, ExtractLLM
import os 
import argparse
import pandas as pd
import csv
from tqdm import tqdm  
from Rerank import Reranker

def process(room_folder, input_dir, input_text, input_pano, maskcreation, zeropainter, faissmanager):
    if not input_text:
        print('Cannot read text')
    if not input_pano:
        print('Cannot read image')
    
    mask, check = maskcreation(input_pano)
    

    if mask == None:
        result_text2text, embed_mean_text = faissmanager.search(
            query=input_text, query_type='text', search_in='text',
            top_k=10, return_answer_vector=True)

        result_text2img, embed_text2img = faissmanager.search(
            query=input_text, query_type='text', search_in='image',
            top_k=10, return_answer_vector=True)
        result_image2mean = 0
        result_img2img = 0
        def find_answer():
            score_map = {}
            all_results = result_text2img + result_text2text + result_text2img + result_text2img + result_text2text
            for obj in all_results:
                key = obj['object_dir']
                score_map[key] = score_map.get(key, 0) + obj['confidence']

            sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            top_10_object_dirs = [k for k, v in sorted_items[:10]]
            top_10_object_dirs.insert(0, room_folder)
            top_10_reranked = reranker(input_dir, top_10_object_dirs) 
            return top_10_reranked

    elif mask != None and check==1:
        input_text_extracted = extractor.extracting(input_text)
        output_Zeropainter = zeropainter.ZP(mask, input_text_extracted)
        result_img2img, embed_img = faissmanager.search(
            query=output_Zeropainter, query_type='image', search_in='image',
            top_k=10, return_answer_vector=True)

        result_image2text, embed_mean = faissmanager.search(
            query=output_Zeropainter, query_type='image', search_in='text',
            top_k=10, return_answer_vector=True)

        result_text2mean, embed_mean_text = faissmanager.search(
            query=input_text, query_type='text', search_in='mean pooling images',
            top_k=10, return_answer_vector=True)

        result_text2img, embed_text2img = faissmanager.search(
            query=input_text, query_type='text', search_in='image',
            top_k=10, return_answer_vector=True)
        result_text2text = faissmanager.search(
            query=input_text, query_type='text', search_in='text',
            top_k=10, return_answer_vector=False)

        def find_answer():
            score_map = {}
            all_results = result_img2img + result_text2img + result_text2img + result_text2img  + result_image2text
            for obj in all_results:
                key = obj['object_dir']
                score_map[key] = score_map.get(key, 0) + obj['confidence']

            sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            top_10_object_dirs = [k for k, v in sorted_items[:10]]
            top_10_object_dirs.insert(0, room_folder)
            top_10_reranked = reranker(input_dir, top_10_object_dirs) 
            return top_10_reranked
    elif check == 2:
        input_text_extracted = extractor.extracting(input_text)
        output_Zeropainter = zeropainter.ZP(mask, input_text_extracted)
        result_img2img, embed_img = faissmanager.search(
            query=output_Zeropainter, query_type='image', search_in='image',
            top_k=10, return_answer_vector=True)

        result_text2img, embed_text2img = faissmanager.search(
            query=input_text, query_type='text', search_in='image',
            top_k=10, return_answer_vector=True)

        result_image2mean, embed_mean = faissmanager.search(
            query=output_Zeropainter, query_type='image', search_in='mean pooling images',
            top_k=10, return_answer_vector=True)

        def find_answer():
            score_map = {}
            all_results = result_image2mean  + result_img2img + result_image2mean + result_text2img #+ result_text2img + result_text2img  + result_image2text
            for obj in all_results:
                key = obj['object_dir']
                score_map[key] = score_map.get(key, 0) + obj['confidence']

            sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            top_10_object_dirs = [k for k, v in sorted_items[:10]]
            top_10_object_dirs.insert(0, room_folder)
            top_10_reranked = reranker(input_dir, top_10_object_dirs) 
            return top_10_reranked

        
    return find_answer()

def main( input_dir, maskcreation, zeropainter, faissmanager):
    answers = []
    scenes = input_dir + '/scenes'  
    scene_path = os.path.join(input_dir, scenes)  
    room_folders = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]


    for room_folder in tqdm(room_folders, desc="Processing rooms"):
        answer = []
        room_path = os.path.join(scene_path, room_folder)
        text_path = None
        image_path = None

        for file in os.listdir(room_path):
            if file.endswith('.txt'):
                text_path = os.path.join(room_path, file)
            elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(room_path, file)

        if text_path and image_path:
            with open(text_path, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
            try:
                object_dirs = process(room_folder, input_dir, input_text, image_path, maskcreation, zeropainter, faissmanager)
                answer.extend(object_dirs)
                answers.append(answer)
            except ValueError as e:
                print(f"[ERROR] In room '{room_folder}': {e}")
        else:
            print(f"[WARNING] No .txt and/or image in '{room_folder}', skipping...")

    return answers

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run pipeline')
    parser.add_argument('--input', required=True, help='Input dir')
    parser.add_argument('--index_save_dir', required=True, help='Input dir')
    arg = parser.parse_args()
    maskcreation = Maskcreation()
    zeropainter = Inpainting()
    faissmanager = FAISSManager(index_dir=arg.index_save_dir, embedding_dim = 1024)
    extractor = ExtractLLM()
    reranker = Reranker()    
    answer = main(arg.input, maskcreation, zeropainter, faissmanager)
    df = pd.DataFrame(answer)
    df.to_csv('MealsRetrieval_1.csv', index=False)
    print(answer)
