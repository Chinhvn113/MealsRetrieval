import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, set_seed
import numpy as np
import random
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility
    set_seed(seed)  # Transformers library seed

# Set the seed
set_all_seeds(42)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model():
    path = 'OpenGVLab/InternVL2_5-4B-MPO'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=10, do_sample=False, temperature=0.1)
    return model, tokenizer, generation_config


def main(model, tokenizer, generation_config, images_list, Rerank_list, query, query_id):
    # multi-image multi-round conversation, combined images
    pixel_values1 = load_image(images_list[0], max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = load_image(images_list[1], max_num=12).to(torch.bfloat16).cuda()

    # pixel_Pano = load_image(query_image_path, max_num=12).to(torch.bfloat16).cuda()

    pixel_values = torch.cat(( pixel_values1, pixel_values2), dim=0)

    question = f"""<image>\nThere are two images of furniture or appliances above.

Which one matches this description best: "{query}"?

Focus only on these details:

- Number of parts (like doors, handles, shelves).
- Color (e.g., silver, white, black, wood).
- Type of item (fridge, cabinet, shelf, etc.).
- Its purpose or function (storing food, placing a TV, serving coffee).

Choose the one that matches all of these most correctly.
If an image has the wrong number of doors or is a different type, it should not be chosen.
Pay special attention to unique functional details in the description, such as hidden drawers, separated, swivel, split into parts, foldable, ...
Look for the image that fit with most details
Just answer with "1" or "2". No explanation.""" 
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                history=None, return_history=True)
 
    import re
    match = re.search(r'\d+', response) 
    if match:
        number = int(match.group())
        # print(number)  # Output: 1    

    top1 = int(number)
    if top1 == 2:
        print("Query: ", query_id)
        print("New top 1 is: ", Rerank_list[int(number)])
        tmp = Rerank_list[1]
        Rerank_list[1] = Rerank_list[2]
        Rerank_list[2] = tmp
        
        

    return Rerank_list
