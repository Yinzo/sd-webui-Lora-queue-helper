import os
import json
import copy
import random

import gradio as gr

from modules import sd_samplers, errors, scripts, images
from modules.processing import Processed, process_images
from modules.shared import state, cmd_opts


def get_directories(base_path):
    lora_base_path = os.path.join(cmd_opts.lora_dir, base_path)
    try:
        directories = [d for d in os.listdir(lora_base_path) if os.path.isdir(os.path.join(lora_base_path, d))]
    except:
        directories = []
    return directories

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_json(file_path):
    try:
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Extract the filename without the extension
        filename = file_path.rsplit('/', 1)[-1].split('.')[0]

        # Extract the required fields from the JSON data
        preferred_weight = data.get("preferred weight", "")
        activation_text = data.get("activation text", "")

        # Format the output string
        output = f"<lora:{filename}:{preferred_weight}>, {activation_text},"

        return output

    except Exception as e:
        print(f"An error occurred in Lora queue helper: {e}")


class Script(scripts.Script):
    def title(self):
        return "Apply on every Lora"

    def ui(self, is_img2img):
        def update_dirs(base_path):
            return gr.CheckboxGroup.update(choices=get_directories(base_path), value=[])

        def get_lora(base_path, directories):
            all_loras = []

            for directory in directories:
                # replace "/" with "."
                directory = directory.replace("/", ".")

                safetensor_files = [f for f in os.listdir(os.path.join(base_path, directory)) if f.endswith('.safetensors')]
                all_loras.extend([os.path.splitext(f)[0] for f in safetensor_files])
            
            return all_loras

        def update_loras(base_path, directories):
            all_loras = get_lora(os.path.join(cmd_opts.lora_dir, base_path), directories)
            visible = len(all_loras) > 0
            return gr.CheckboxGroup.update(choices=all_loras, value=all_loras, visible=visible), gr.Button.update(visible=visible), gr.Button.update(visible=visible)

        def select_all_dirs(base_dir):
            all_dirs = ["/"] + get_directories(base_dir)
            return gr.CheckboxGroup.update(value=all_dirs)

        def deselect_all_dirs():
            return gr.CheckboxGroup.update(value=[])

        def select_all_lora(all_loras):
            return gr.CheckboxGroup.update(value=all_loras)

        def deselect_all_lora():
            return gr.CheckboxGroup.update(value=[])

        base_dir_textbox = gr.Textbox(label="Base directory", elem_id=self.elem_id("base_dir_textbox"))
        all_dirs = get_directories(base_dir_textbox.value)
        all_dirs.insert(0, "/")

        with gr.Group():
            directory_checkboxes = gr.CheckboxGroup(label="Select Directory", choices=all_dirs, value=["/"], elem_id=self.elem_id("directory_checkboxes"))
            with gr.Row():
                select_all_dirs_button = gr.Button("All", size="sm")
                deselect_all_dirs_button = gr.Button("Clear", size="sm")

        startup_loras = get_lora(base_dir_textbox.value, directory_checkboxes.value)
        
        with gr.Group():
            tags_checkboxes = gr.CheckboxGroup(label="Lora", choices=startup_loras, value=startup_loras, visible=len(startup_loras)>0, elem_id=self.elem_id("tags_checkboxes"))
            with gr.Row():
                select_all_lora_button = gr.Button("All", size="sm", visible=len(startup_loras)>0)
                deselect_all_lora_button = gr.Button("Clear", size="sm", visible=len(startup_loras)>0)

        checkbox_iterate = gr.Checkbox(label="Use consecutive seed", value=False, elem_id=self.elem_id("checkbox_iterate"))
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
        checkbox_save_grid = gr.Checkbox(label="Save grid image", value=True, elem_id=self.elem_id("checkbox_save_grid"))
       
        base_dir_textbox.change(fn=update_dirs, inputs=base_dir_textbox, outputs=[directory_checkboxes])
        directory_checkboxes.change(fn=update_loras, inputs=[base_dir_textbox, directory_checkboxes], outputs=[tags_checkboxes, select_all_lora_button, deselect_all_lora_button])
        select_all_lora_button.click(fn=select_all_lora, inputs=tags_checkboxes, outputs=tags_checkboxes)
        deselect_all_lora_button.click(fn=deselect_all_lora, inputs=None, outputs=tags_checkboxes)
        select_all_dirs_button.click(fn=select_all_dirs, inputs=base_dir_textbox, outputs=directory_checkboxes)
        deselect_all_dirs_button.click(fn=deselect_all_dirs, inputs=None, outputs=directory_checkboxes)

        return [base_dir_textbox, directory_checkboxes, tags_checkboxes, checkbox_iterate, checkbox_iterate_batch, checkbox_save_grid]

    def run(self, p, base_path, directories, selected_files, checkbox_iterate, checkbox_iterate_batch, is_save_grid):
        p.do_not_save_grid = not is_save_grid

        job_count = 0
        jobs = []

        for directory in directories:
            safetensor_files = [f for f in os.listdir(os.path.join(cmd_opts.lora_dir, base_path, directory)) if f.endswith('.safetensors')]
            
            for safetensor_file in safetensor_files:
                lora_filename = os.path.splitext(safetensor_file)[0]
                if lora_filename not in selected_files:
                    continue
                json_file = lora_filename + '.json'
                json_file_path = os.path.join(base_path, directory, json_file)
                
                if os.path.exists(json_file_path):
                    additional_prompt = process_json(json_file_path)
                else:
                    additional_prompt = f"<lora:{lora_filename}:1>,"

                args = {}
                args["prompt"] = additional_prompt +"," + p.prompt

                job_count += args.get("n_iter", p.n_iter)

                jobs.append(args)

        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        result_images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            result_images += proc.images

            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        if is_save_grid:
            grid_image = images.image_grid(result_images, rows=1)
            result_images.insert(0, grid_image)


        return Processed(p, result_images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
