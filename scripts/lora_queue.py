import os
import json
import copy
import random

import gradio as gr

from modules import sd_samplers, errors, scripts, images, sd_models
from modules.processing import Processed, process_images
from modules.shared import state, cmd_opts, opts
from pathlib import Path

lora_dir = Path(cmd_opts.lora_dir).resolve()


def allowed_path(path):
    return Path(path).resolve().is_relative_to(lora_dir)


def get_base_path(is_use_custom_path, custom_path):
    return lora_dir.joinpath(custom_path) if is_use_custom_path else lora_dir


def get_directories(base_path):
    directories = ["/"]
    try:
        if allowed_path(base_path):
            directories.extend([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)
    return directories

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_lora_prompt(lora_path, json_path):
    try:
        # Open and read the JSON file
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Extract the required fields from the JSON data
        preferred_weight = data.get("preferred weight", "1")
        activation_text = data.get("activation text", "")

        if opts.lora_preferred_name == "Filename":
            lora_name = lora_path.stem
        else:
            metadata = sd_models.read_metadata_from_safetensors(lora_path)
            lora_name = metadata.get('ss_output_name', lora_path.stem)

        # Format the prompt string
        output = f"<lora:{lora_name}:{preferred_weight}>, {activation_text},"

        return output

    except Exception as e:
        print(f"An error occurred in Lora queue helper: {e}")


class Script(scripts.Script):
    def title(self):
        return "Apply on every Lora"

    def ui(self, is_img2img):
        def update_dirs(is_use_custom_path, custom_path):
            base_path = get_base_path(is_use_custom_path, custom_path)
            dirs = get_directories(base_path)
            return gr.CheckboxGroup.update(choices=dirs, value=[])

        def show_dir_textbox(enabled, custom_path):
            all_dirs = get_directories(lora_dir.joinpath(custom_path) if enabled else lora_dir)
            return gr.Textbox.update(visible=enabled), gr.CheckboxGroup.update(choices=all_dirs, value=[])

        def get_lora(base_path, directories):
            all_loras = []

            for directory in directories:
                # if directory is "/" use base_path
                directory = base_path if directory == "/" else os.path.join(base_path, directory)
                if not allowed_path(directory):
                    continue
                safetensor_files = [f for f in os.listdir(directory) if f.endswith('.safetensors')]
                all_loras.extend([os.path.splitext(f)[0] for f in safetensor_files])

            return all_loras

        def update_loras(is_use_custom_path, custom_path, directories):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_loras = get_lora(base_path, directories)
            visible = len(all_loras) > 0
            return gr.CheckboxGroup.update(choices=all_loras, value=all_loras, visible=visible), gr.Button.update(
                visible=visible), gr.Button.update(visible=visible)

        def select_all_dirs(is_use_custom_path, custom_path):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_dirs = get_directories(base_path)
            return gr.CheckboxGroup.update(value=all_dirs)

        def deselect_all_dirs():
            return gr.CheckboxGroup.update(value=[])

        def select_all_lora(is_use_custom_path, custom_path, directories):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_loras = get_lora(base_path, directories)
            return gr.CheckboxGroup.update(value=all_loras)

        def deselect_all_lora():
            return gr.CheckboxGroup.update(value=[])

        base_dir_checkbox = gr.Checkbox(label="Use Custom Lora path", value=False,
                                        elem_id=self.elem_id("base_dir_checkbox"))
        base_dir_textbox = gr.Textbox(label="Lora directory", placeholder="Relative path under Lora directory. Use --lora-dir to set Lora directory at WebUI startup.", visible=False, elem_id=self.elem_id("base_dir_textbox"))
        base_dir = base_dir_textbox.value if base_dir_checkbox.value else lora_dir
        all_dirs = get_directories(base_dir)

        with gr.Group():
            directory_checkboxes = gr.CheckboxGroup(label="Select Directory", choices=all_dirs, value=["/"], elem_id=self.elem_id("directory_checkboxes"))
            with gr.Row():
                select_all_dirs_button = gr.Button("All", size="sm")
                deselect_all_dirs_button = gr.Button("Clear", size="sm")

        startup_loras = get_lora(base_dir, directory_checkboxes.value)
        
        with gr.Group():
            lora_checkboxes = gr.CheckboxGroup(label="Lora", choices=startup_loras, value=startup_loras, visible=len(startup_loras)>0, elem_id=self.elem_id("lora_checkboxes"))
            with gr.Row():
                select_all_lora_button = gr.Button("All", size="sm", visible=len(startup_loras)>0)
                deselect_all_lora_button = gr.Button("Clear", size="sm", visible=len(startup_loras)>0)

        checkbox_iterate = gr.Checkbox(label="Use consecutive seed", value=False, elem_id=self.elem_id("checkbox_iterate"))
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
        checkbox_save_grid = gr.Checkbox(label="Save grid image", value=True, elem_id=self.elem_id("checkbox_save_grid"))

        base_dir_checkbox.change(fn=show_dir_textbox, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[base_dir_textbox, directory_checkboxes])
        base_dir_textbox.change(fn=update_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[directory_checkboxes])
        directory_checkboxes.change(fn=update_loras, inputs=[base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button])
        select_all_lora_button.click(fn=select_all_lora, inputs=[base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=lora_checkboxes)
        deselect_all_lora_button.click(fn=deselect_all_lora, inputs=None, outputs=lora_checkboxes)
        select_all_dirs_button.click(fn=select_all_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=directory_checkboxes)
        deselect_all_dirs_button.click(fn=deselect_all_dirs, inputs=None, outputs=directory_checkboxes)

        return [base_dir_checkbox, base_dir_textbox, directory_checkboxes, lora_checkboxes, checkbox_iterate, checkbox_iterate_batch, checkbox_save_grid]

    def run(self, p, is_use_custom_path, custom_path, directories, selected_loras, checkbox_iterate, checkbox_iterate_batch, is_save_grid):
        p.do_not_save_grid = not is_save_grid

        job_count = 0
        jobs = []

        base_path = get_base_path(is_use_custom_path, custom_path)
        for directory in directories:
            # if directory is "/" use base_path
            directory = base_path if directory == "/" else base_path.joinpath(directory)
            if not allowed_path(directory):
                continue
            safetensor_files = [f for f in os.listdir(directory) if f.endswith('.safetensors')]

            for safetensor_file in safetensor_files:
                lora_filename = os.path.splitext(safetensor_file)[0]
                if lora_filename not in selected_loras:
                    continue
                lora_file_path = directory.joinpath(safetensor_file)
                json_file = lora_filename + '.json'
                json_file_path = directory.joinpath(json_file)

                if os.path.exists(json_file_path):
                    additional_prompt = get_lora_prompt(lora_file_path, json_file_path)
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

        if is_save_grid and len(result_images) > 1:
            grid_image = images.image_grid(result_images, rows=1)
            result_images.insert(0, grid_image)
            all_prompts.insert(0, "")
            infotexts.insert(0, "")


        return Processed(p, result_images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
