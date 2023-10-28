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


def load_and_save_tags(lora_name, print_tags, query_tags, force_fetch):
    json_tags_path = "./loras_tags.json"
    lora_tags = load_json_from_file(json_tags_path)
    output_tags_list = lora_tags.get(lora_name, None) if lora_tags is not None else None
    if output_tags_list is not None:
        output_tags = ", ".join(output_tags_list)
        if print_tags:
                print("trainedWords:",output_tags)
    else:
        output_tags_list = []
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
                output_tags_list = model_info["trainedWords"]
                if print_tags:
                    print("trainedWords:",output_tags)
        else:
            print("No informations found.")
            if lora_tags is None:
                    lora_tags = {}
            lora_tags[lora_name] = []
            save_dict_to_json(lora_tags,json_tags_path)

    return lora_path, output_tags, output_tags_list

class LoraLoaderTagsQuery:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (LORA_LIST, ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
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
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "LIST",)
    RETURN_NAMES = ("MODEL", "CLIP", "civitai_tags", "civitai_tags_list")
    FUNCTION = "load_lora"
    CATEGORY = "llwt"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, query_tags, tags_out, print_tags, bypass, force_fetch, opt_prompt=None):
        if strength_model == 0 and strength_clip == 0 or bypass:
            if opt_prompt is not None:
                out_string = opt_prompt
            else:
                out_string = ""
            return (model, clip, out_string,)
        
        
        lora_path, output_tags, output_tags_list = load_and_save_tags(lora_name, print_tags, query_tags, force_fetch)

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
                output_tags_list.append(opt_prompt)
            else:
                output_tags = opt_prompt
                output_tags_list = [opt_prompt]
        return (model_lora, clip_lora, output_tags, output_tags_list,)
    

class LoraTagsQueryOnly:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": { 
                "lora_name": (LORA_LIST, ),
                "query_tags": ("BOOLEAN", {"default": True}),
                "tags_out": ("BOOLEAN", {"default": True}),
                "print_tags": ("BOOLEAN", {"default": False}),
                "force_fetch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "opt_prompt": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "LIST",)
    RETURN_NAMES = ("civitai_tags", "civitai_tags_list")
    FUNCTION = "load_lora"
    CATEGORY = "llwt"

    def load_lora(self, lora_name, query_tags, tags_out, print_tags, force_fetch, opt_prompt=None):
        _, output_tags, output_tags_list = load_and_save_tags(lora_name, print_tags, query_tags, force_fetch)

        if opt_prompt is not None:
            if tags_out:
                output_tags = opt_prompt+", "+output_tags
                output_tags_list.append(opt_prompt)
            else:
                output_tags = opt_prompt
                output_tags_list = [opt_prompt]
        return (output_tags, output_tags_list,)
    
    
class TagsSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "tags_list": ("LIST", {"default": []}),
                "selector": ("STRING", {"default": ":"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_tags"
    CATEGORY = "llwt"

    def select_tags(self, tags_list, selector):
        range_index_list = selector.split(",")
        output = {}
        for range_index in range_index_list:
            # single value
            if range_index.count(":") == 0:
                index = int(range_index)
                output[index] = tags_list[index]

            # actual range
            if range_index.count(":") == 1:
                indexes = range_index.split(":")
                # check empty
                if indexes[0] == "":
                    start = 0
                else:
                    start = int(indexes[0])
                if indexes[1] == "":
                    end = len(tags_list)
                else:
                    end = int(indexes[1])
                # check negative
                if start < 0:
                    start = len(tags_list) + start
                if end < 0:
                    end = len(tags_list) + end
                # merge all
                for i in range(start, end):
                    output[i] = tags_list[i]
        output_tags = ", ".join(list(output.values()))

        return (output_tags,)

class TagsViewer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "tags_list": ("LIST", {"default": []}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "format_tags"
    CATEGORY = "llwt"

    def format_tags(self, tags_list):
        output = ""
        i = 0
        for tag in tags_list:
            output += f'{i} : "{tag}"\n'
            i+=1

        return (output,)
    
NODE_CLASS_MAPPINGS = {
    "LoraLoaderTagsQuery": LoraLoaderTagsQuery,
    "LoraTagsQueryOnly": LoraTagsQueryOnly,
    "TagsSelector": TagsSelector,
    "TagsViewer": TagsViewer,
}
