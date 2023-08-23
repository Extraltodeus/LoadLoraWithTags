import folder_paths
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file
import hashlib
import requests
import json

def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None

def save_dict_to_json(data_dict, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
            print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to file: {e}")

def get_model_version_info(hash_value):
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class LoraLoaderTagsQuery:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (LORA_LIST, ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "query_tags": ("BOOLEAN", {"default": True}),
                              "tags_out": ("BOOLEAN", {"default": True}),
                              "print_tags": ("BOOLEAN", {"default": False}),
                              "bypass": ("BOOLEAN", {"default": False}),
                              "force_fetch": ("BOOLEAN", {"default": False}),
                              },
                "optional":
                            {
                                "opt_prompt": ("STRING", {"forceInput": True}),
                            }
                }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, query_tags, tags_out, print_tags, bypass, force_fetch, opt_prompt=None):
        if strength_model == 0 and strength_clip == 0 or bypass:
            if opt_prompt is not None:
                out_string = opt_prompt
            else:
                out_string = ""
            return (model, clip, out_string,)
        
        json_tags_path = "./loras_tags.json"
        lora_tags = load_json_from_file(json_tags_path)
        output_tags = lora_tags.get(lora_name, None) if lora_tags is not None else None
        if output_tags is not None:
            output_tags = ", ".join(output_tags)
            if print_tags:
                    print("trainedWords:",output_tags)
        else:
            output_tags = ""

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if (query_tags and output_tags == "") or force_fetch:
            print("calculating lora hash")
            LORAsha256 = calculate_sha256(lora_path)
            print("requesting infos")
            model_info = get_model_version_info(LORAsha256)
            if model_info is not None:
                if "trainedWords" in model_info:
                    print("tags found!")
                    if lora_tags is None:
                        lora_tags = {}
                    lora_tags[lora_name] = model_info["trainedWords"]
                    save_dict_to_json(lora_tags,json_tags_path)
                    output_tags = ", ".join(model_info["trainedWords"])
                    if print_tags:
                        print("trainedWords:",output_tags)
            else:
                print("No informations found.")
                if lora_tags is None:
                        lora_tags = {}
                lora_tags[lora_name] = []
                save_dict_to_json(lora_tags,json_tags_path)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        if opt_prompt is not None:
            if tags_out:
                output_tags = opt_prompt+", "+output_tags
            else:
                output_tags = opt_prompt
        return (model_lora, clip_lora, output_tags,)
    
NODE_CLASS_MAPPINGS = {
    "LoraLoaderTagsQuery": LoraLoaderTagsQuery,
}
