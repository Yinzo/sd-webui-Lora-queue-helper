import os
import json
import copy
import random
import math

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

def is_directory_contain_lora(path):
    try:
        if allowed_path(path):
            safetensor_files = [f for f in os.listdir(path) if f.endswith('.safetensors')]
            return len(safetensor_files) > 0
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)

    return False

def get_directories(base_path, include_root=True):
    directories = ["/"] if include_root else []
    try:
        if allowed_path(base_path):
            for entry in os.listdir(base_path):
                full_path = os.path.join(base_path, entry)
                if os.path.isdir(full_path):
                    if is_directory_contain_lora(full_path):
                        directories.append(entry)
                    
                    nested_directories = get_directories(full_path, include_root=False)
                    directories.extend([os.path.join(entry, d) for d in nested_directories])

    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)

    return directories

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_lora_prompt(lora_path, json_path):
    # Open and read the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract the required fields from the JSON data
    preferred_weight = data.get("preferred weight", 1)
    activation_text = data.get("activation text", "")

    try:
        if float(preferred_weight) == 0:
            preferred_weight = 1
    except:
        preferred_weight = 1

    if opts.lora_preferred_name == "Filename":
        lora_name = lora_path.stem
    else:
        metadata = sd_models.read_metadata_from_safetensors(lora_path)
        lora_name = metadata.get('ss_output_name', lora_path.stem)

    # Format the prompt string
    output = f"<lora:{lora_name}:{preferred_weight}>, {activation_text},"

    return output

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

        def update_loras(current_selected, is_use_custom_path, custom_path, directories):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_loras = get_lora(base_path, directories)
            visible = len(all_loras) > 0
            new_values = [lora for lora in all_loras if lora in current_selected]
            return gr.CheckboxGroup.update(choices=all_loras, value=new_values, visible=visible), gr.Button.update(
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

        def toggle_row_number(checked):
            return gr.Number.update(visible=checked), gr.Checkbox.update(visible=checked)

        def toggle_auto_row_number(checked):
            return gr.Number.update(interactive=not checked)

        with gr.Column():
            base_dir_checkbox = gr.Checkbox(label="Use Custom Lora path", value=False,
                                            elem_id=self.elem_id("base_dir_checkbox"))
            base_dir_textbox = gr.Textbox(label="Lora directory", placeholder="Relative path under Lora directory. Use --lora-dir to set Lora directory at WebUI startup.", visible=False, elem_id=self.elem_id("base_dir_textbox"))
            base_dir = base_dir_textbox.value if base_dir_checkbox.value else lora_dir
            all_dirs = get_directories(base_dir)

            directory_checkboxes = gr.CheckboxGroup(label="Select Directory", choices=all_dirs, value=["/"], elem_id=self.elem_id("directory_checkboxes"))

            with gr.Row():
                select_all_dirs_button = gr.Button("All")
                deselect_all_dirs_button = gr.Button("Clear")

            startup_loras = get_lora(base_dir, directory_checkboxes.value)
            
            lora_checkboxes = gr.CheckboxGroup(label="Lora", choices=startup_loras, value=startup_loras, visible=len(startup_loras)>0, elem_id=self.elem_id("lora_checkboxes"))

            with gr.Row():
                select_all_lora_button = gr.Button("All", visible=len(startup_loras)>0)
                deselect_all_lora_button = gr.Button("Clear", visible=len(startup_loras)>0)

            with gr.Row():
                checkbox_iterate = gr.Checkbox(label="Use consecutive seed", value=False, elem_id=self.elem_id("checkbox_iterate"))
                checkbox_iterate_batch = gr.Checkbox(label="Use same random seed", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
            
            with gr.Row(equal_height=True):
                with gr.Column():
                    checkbox_save_grid = gr.Checkbox(label="Save grid image", value=True, elem_id=self.elem_id("checkbox_save_grid"))
                    checkbox_auto_row_number = gr.Checkbox(label="Auto row number", value=True, elem_id=self.elem_id("checkbox_auto_row_number"))
                grid_row_number = gr.Number(label="Grid row number", value=1, interactive=False, elem_id=self.elem_id("grid_row_number"))

            base_dir_checkbox.change(fn=show_dir_textbox, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[base_dir_textbox, directory_checkboxes])
            base_dir_textbox.change(fn=update_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[directory_checkboxes])
            directory_checkboxes.change(fn=update_loras, inputs=[lora_checkboxes, base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button])
            select_all_lora_button.click(fn=select_all_lora, inputs=[base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=lora_checkboxes)
            deselect_all_lora_button.click(fn=deselect_all_lora, inputs=None, outputs=lora_checkboxes)
            select_all_dirs_button.click(fn=select_all_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=directory_checkboxes)
            deselect_all_dirs_button.click(fn=deselect_all_dirs, inputs=None, outputs=directory_checkboxes)
            checkbox_save_grid.change(fn=toggle_row_number, inputs=checkbox_save_grid, outputs=[grid_row_number, checkbox_auto_row_number])
            checkbox_auto_row_number.change(fn=toggle_auto_row_number, inputs=[checkbox_auto_row_number], outputs=grid_row_number)

        return [base_dir_checkbox, base_dir_textbox, directory_checkboxes, lora_checkboxes, checkbox_iterate, checkbox_iterate_batch, checkbox_save_grid, checkbox_auto_row_number, grid_row_number]

    def run(self, p, is_use_custom_path, custom_path, directories, selected_loras, checkbox_iterate, checkbox_iterate_batch, is_save_grid, is_auto_row_number, row_number):
        if len(selected_loras) == 0:
            return process_images(p)

        p.do_not_save_grid = True  # disable default grid image

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

                additional_prompt = None
                if os.path.exists(json_file_path):
                    try:
                        additional_prompt = get_lora_prompt(lora_file_path, json_file_path)
                    except Exception as e:
                        print(f"Lora Queue Helper got error when loading lora info, error: {e}")
                
                if additional_prompt == None or not isinstance(additional_prompt, str):
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
            if is_auto_row_number:
                # get a 4:3 rectangular width
                row_number = round(3.0 * math.sqrt(len(result_images)/12.0))
            else:
                row_number = int(row_number)

            grid_image = images.image_grid(result_images, rows=row_number)
            result_images.insert(0, grid_image)
            all_prompts.insert(0, "")
            infotexts.insert(0, "")

        return Processed(p, result_images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
