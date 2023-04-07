import html
import os

import torch
import transformers

from modules import shared

from modules import scripts, devices



class Model:
    name = None
    model = None
    tokenizer = None


available_models = []
current = Model()

base_dir = scripts.basedir()
models_dir = os.path.join(base_dir, "models")


def device():
    return devices.cpu if shared.opts.promptgen_device == "cpu" else devices.device


def list_available_models():
    available_models.clear()

    os.makedirs(models_dir, exist_ok=True)

    for dirname in os.listdir(models_dir):
        if os.path.isdir(os.path.join(models_dir, dirname)):
            available_models.append(dirname)

    for name in [x.strip() for x in shared.opts.promptgen_names.split(",")]:
        if not name:
            continue

        available_models.append(name)


def get_model_path(name):
    dirname = os.path.join(models_dir, name)
    if not os.path.isdir(dirname):
        return name

    return dirname


def generate_batch(
    input_ids,
    min_length = 20,
    max_length = 150,
    num_beams = 1,
    temperature = 1,
    repetition_penalty = 1,
    length_penalty = 1,
    sampling_mode = "Top K",
    top_k = 12,
    top_p = 0.15,
):
    top_p = float(top_p) if sampling_mode == "Top P" else None
    top_k = int(top_k) if sampling_mode == "Top K" else None

    outputs = current.model.generate(
        input_ids,
        do_sample=True,
        temperature=max(float(temperature), 1e-6),
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        top_p=top_p,
        top_k=top_k,
        num_beams=int(num_beams),
        min_length=min_length,
        max_length=max_length,
        pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id,
    )
    texts = current.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


def generate_magic_prompt(text, batch_size = 4, *args):
    model_name = "Gustavosta/MagicPrompt-Stable-Diffusion"
    batch_count = 1
    shared.state.textinfo = "Loading model..."
    shared.state.job_count = batch_count

    if current.name != model_name:
        current.tokenizer = None
        current.model = None
        current.name = None

        if model_name != "None":
            path = get_model_path(model_name)
            current.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            current.model = transformers.AutoModelForCausalLM.from_pretrained(path)
            current.name = model_name

    assert current.model, "No model available"
    assert current.tokenizer, "No tokenizer available"

    current.model.to(device())

    shared.state.textinfo = ""

    input_ids = current.tokenizer(text, return_tensors="pt").input_ids
    if input_ids.shape[1] == 0:
        input_ids = torch.asarray([[current.tokenizer.bos_token_id]], dtype=torch.long)
    input_ids = input_ids.to(device())
    input_ids = input_ids.repeat((batch_size, 1))
    output = []
    for i in range(batch_count):
        texts = generate_batch(input_ids, *args)
        shared.state.nextjob()
        for generated_text in texts:
            output.append(generated_text);

    return output
