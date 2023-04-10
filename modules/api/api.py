import base64
import io
import os
import time
import rembg
import cgi
import urllib
import datetime
import uvicorn
import gradio as gr
from threading import Lock
from io import BytesIO
from gradio.processing_utils import decode_base64_to_file
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import numpy as np
import imutils
import cv2

import modules.shared as shared
from modules import (
    sd_samplers,
    deepbooru,
    sd_hijack,
    images,
    scripts,
    ui,
    postprocessing,
)
from modules.interrogate import image_to_prompt
from modules.promptgen import generate_magic_prompt
from modules.api.models import *
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    process_images,
)
from modules.textual_inversion.textual_inversion import (
    create_embedding,
    train_embedding,
)
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin, Image
from modules.sd_models import (
    checkpoints_list,
    checkpoints_loaded,
    unload_model_weights,
    reload_model_weights,
)
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from modules.paths_internal import models_path, embeddings_path
from typing import Callable, List
import piexif
import piexif.helper

# Binding in pixelization extension
pixelization_fn: Callable[[object, int, bool], object] = None

# Available lora
list_lora_fn: Callable[[], dict] = None

# Fix download 403
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)


def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in sd_upscalers])}",
        )


def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found")


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict["extras_upscaler_1"] = reqDict.pop("upscaler_1", None)
    reqDict["extras_upscaler_2"] = reqDict.pop("upscaler_2", None)
    return reqDict


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        raise HTTPException(status_code=500, detail="Invalid encoded image")


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        if opts.samples_format.lower() == "png":
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(
                output_bytes,
                format="PNG",
                pnginfo=(metadata if use_metadata else None),
                quality=opts.jpeg_quality,
            )

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get("parameters", None)
            exif_bytes = piexif.dump(
                {
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                            parameters or "", encoding="unicode"
                        )
                    }
                }
            )
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(
                    output_bytes,
                    format="JPEG",
                    exif=exif_bytes,
                    quality=opts.jpeg_quality,
                )
            else:
                image.save(
                    output_bytes,
                    format="WEBP",
                    exif=exif_bytes,
                    quality=opts.jpeg_quality,
                )

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def api_middleware(app: FastAPI):
    rich_available = True
    try:
        import anyio  # importing just so it can be placed on silent list
        import starlette  # importing just so it can be placed on silent list
        from rich.console import Console

        console = Console()
    except:
        import traceback

        rich_available = False

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get("path", "err")
        if shared.cmd_opts.api_log and endpoint.startswith("/sdapi"):
            print(
                "API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}".format(
                    t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    code=res.status_code,
                    ver=req.scope.get("http_version", "0.0"),
                    cli=req.scope.get("client", ("0:0.0.0", 0))[0],
                    prot=req.scope.get("scheme", "err"),
                    method=req.scope.get("method", "err"),
                    endpoint=endpoint,
                    duration=duration,
                )
            )
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        print(f"API error: {request.method}: {request.url} {err}")
        if not isinstance(
            e, HTTPException
        ):  # do not print backtrace on known httpexceptions
            if rich_available:
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                traceback.print_exc()
        return JSONResponse(
            status_code=vars(e).get("status_code", 500), content=jsonable_encoder(err)
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        if shared.cmd_opts.api_auth:
            self.credentials = dict()
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        api_middleware(self.app)
        self.add_api_route(
            "/sdapi/v1/txt2img",
            self.text2imgapi,
            methods=["POST"],
            response_model=TextToImageResponse,
        )
        self.add_api_route(
            "/sdapi/v1/img2img",
            self.img2imgapi,
            methods=["POST"],
            response_model=ImageToImageResponse,
        )
        self.add_api_route(
            "/sdapi/v1/extra-single-image",
            self.extras_single_image_api,
            methods=["POST"],
            response_model=ExtrasSingleImageResponse,
        )
        self.add_api_route(
            "/sdapi/v1/extra-batch-images",
            self.extras_batch_images_api,
            methods=["POST"],
            response_model=ExtrasBatchImagesResponse,
        )
        self.add_api_route(
            "/sdapi/v1/png-info",
            self.pnginfoapi,
            methods=["POST"],
            response_model=PNGInfoResponse,
        )
        self.add_api_route(
            "/sdapi/v1/progress",
            self.progressapi,
            methods=["GET"],
            response_model=ProgressResponse,
        )
        self.add_api_route(
            "/sdapi/v1/interrogate", self.interrogateapi, methods=["POST"]
        )
        self.add_api_route("/sdapi/v1/interrupt", self.interruptapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        self.add_api_route(
            "/sdapi/v1/options",
            self.get_config,
            methods=["GET"],
            response_model=OptionsModel,
        )
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        self.add_api_route(
            "/sdapi/v1/cmd-flags",
            self.get_cmd_flags,
            methods=["GET"],
            response_model=FlagsModel,
        )
        self.add_api_route(
            "/sdapi/v1/samplers",
            self.get_samplers,
            methods=["GET"],
            response_model=List[SamplerItem],
        )
        self.add_api_route(
            "/sdapi/v1/upscalers",
            self.get_upscalers,
            methods=["GET"],
            response_model=List[UpscalerItem],
        )
        self.add_api_route(
            "/sdapi/v1/sd-models",
            self.get_sd_models,
            methods=["GET"],
            response_model=List[SDModelItem],
        )
        self.add_api_route(
            "/sdapi/v1/hypernetworks",
            self.get_hypernetworks,
            methods=["GET"],
            response_model=List[HypernetworkItem],
        )
        self.add_api_route(
            "/sdapi/v1/face-restorers",
            self.get_face_restorers,
            methods=["GET"],
            response_model=List[FaceRestorerItem],
        )
        self.add_api_route(
            "/sdapi/v1/realesrgan-models",
            self.get_realesrgan_models,
            methods=["GET"],
            response_model=List[RealesrganItem],
        )
        self.add_api_route(
            "/sdapi/v1/prompt-styles",
            self.get_prompt_styles,
            methods=["GET"],
            response_model=List[PromptStyleItem],
        )
        self.add_api_route(
            "/sdapi/v1/embeddings",
            self.get_embeddings,
            methods=["GET"],
            response_model=EmbeddingsResponse,
        )
        self.add_api_route(
            "/sdapi/v1/lora",
            self.get_lora,
            methods=["GET"],
            response_model=LorasResponse,
        )
        self.add_api_route(
            "/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"]
        )
        self.add_api_route(
            "/sdapi/v1/create/embedding",
            self.create_embedding,
            methods=["POST"],
            response_model=CreateResponse,
        )
        self.add_api_route(
            "/sdapi/v1/create/hypernetwork",
            self.create_hypernetwork,
            methods=["POST"],
            response_model=CreateResponse,
        )
        self.add_api_route(
            "/sdapi/v1/preprocess",
            self.preprocess,
            methods=["POST"],
            response_model=PreprocessResponse,
        )
        self.add_api_route(
            "/sdapi/v1/train/embedding",
            self.train_embedding,
            methods=["POST"],
            response_model=TrainResponse,
        )
        self.add_api_route(
            "/sdapi/v1/train/hypernetwork",
            self.train_hypernetwork,
            methods=["POST"],
            response_model=TrainResponse,
        )
        self.add_api_route(
            "/sdapi/v1/memory",
            self.get_memory,
            methods=["GET"],
            response_model=MemoryResponse,
        )
        self.add_api_route(
            "/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"]
        )
        self.add_api_route(
            "/sdapi/v1/reload-checkpoint", self.reloadapi, methods=["POST"]
        )
        self.add_api_route(
            "/sdapi/v1/scripts",
            self.get_scripts_list,
            methods=["GET"],
            response_model=ScriptsList,
        )
        self.add_api_route(
            "/sdapi/v1/rm-background",
            self.rm_background,
            methods=["POST"],
            response_model=RemoveBackgroundResponse,
        )
        self.add_api_route(
            "/sdapi/v1/img2text",
            self.img2text,
            methods=["POST"],
            response_model=Img2TextResponse,
        )
        self.add_api_route(
            "/sdapi/v1/prompt",
            self.prompt_gen,
            methods=["POST"],
            response_model=PromptGenResponse,
        )
        self.add_api_route(
            "/sdapi/v1/pixelize",
            self.pixelize,
            methods=["POST"],
            response_model=PixelizeResponse,
        )
        self.add_api_route(
            "/sdapi/v1/pose/face-mask",
            self.face_mask,
            methods=["POST"],
            response_model=FaceMaskResponse,
        )
        self.add_api_route(
            "/sdapi/v1/download-lora",
            self.download_lora,
            methods=["POST"],
            response_model=DownloadPluginsResponse,
        )
        self.add_api_route(
            "/sdapi/v1/download-embedding",
            self.download_embedding,
            methods=["POST"],
            response_model=DownloadPluginsResponse,
        )


        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    async def download_lora(self, input: DownloadPluginsResquest):
        r = urllib.request.urlopen(input.url)
        stuff = r.info()['Content-Disposition']
        value, params = cgi.parse_header(stuff)
        file_name = params["filename"]
        path = os.path.join(models_path, "Lora", file_name)
        if os.path.exists(path):
            return {"success": False, "file_name": file_name}
        urllib.request.urlretrieve(input.url, path)
        return {"success": True, "file_name": file_name}
        
    async def download_embedding(self, input: DownloadPluginsResquest):
        r = urllib.request.urlopen(input.url)
        stuff = r.info()['Content-Disposition']
        value, params = cgi.parse_header(stuff)
        file_name = params["filename"]
        path = os.path.join(embeddings_path, file_name)
        if os.path.exists(path):
            return {"success": False, "file_name": file_name}
        urllib.request.urlretrieve(input.url, path)
        return {"success": True, "file_name": file_name}


    async def get_lora(self):
        if list_lora_fn != None:
            loaded = []
            loras = list_lora_fn()
            for i in loras:
                loaded.append(loras[i].name)
            return {"loaded": loaded}
        return {"loaded": {}}

    async def face_mask(self, input: FaceMaskResquest):
        templateB64 = "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"
        img = base64.b64decode(input.image)
        tpl = base64.b64decode(templateB64)
        np_image_o = np.fromstring(img, dtype=np.uint8)
        np_template = np.fromstring(tpl, dtype=np.uint8)

        image_o = cv2.imdecode(np_image_o, cv2.IMREAD_COLOR)
        template = cv2.imdecode(np_template, cv2.IMREAD_COLOR)

        loc = False
        threshold = 0.5
        x, w, h = template.shape[::-1]
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(template, width=int(template.shape[1] * scale))
            x, w, h = resized.shape[::-1]
            res = cv2.matchTemplate(image_o, resized, cv2.TM_CCORR_NORMED)

            loc = np.where(res >= threshold)
            if len(list(zip(*loc[::-1]))) > 0:
                break

        target_center_x = 0
        target_center_y = 0

        if loc and len(list(zip(*loc[::-1]))) > 0:
            for pt in zip(*loc[::-1]):
                center = (pt[0] + w // 2, pt[1] + h // 2)
                target_center_x = center[0]
                target_center_y = center[1]
                # cv2.rectangle(image_o, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                # cv2.rectangle(input, (pt[0]*2, pt[1]*2), ((pt[0] + w)*2, (pt[1] + h)*2), (0, 0, 255), 2)
                image_o.fill(0)
                cv2.ellipse(image_o, center, (w, h), 0, 0, 360, (255, 255, 255), -1)

        retval, buffer = cv2.imencode(".jpg", image_o)
        jpg_as_text = base64.b64encode(buffer)

        return {
            "output": jpg_as_text,
            "center_x": target_center_x,
            "center_y": target_center_y,
        }

    def pixelize(self, req: PixelizeResquest):
        global pixelization_fn
        image = decode_base64_to_image(req.image)
        output = pixelization_fn(image, req.size)
        return {"output": encode_pil_to_base64(output)}

    def prompt_gen(self, req: PromptGenResquest):
        output = generate_magic_prompt(req.input, req.count)
        return {"output": output}

    def img2text(self, input: Img2TextResquest):
        image = decode_base64_to_image(input.image)
        output = image_to_prompt(image)
        return {"caption": output}

    def rm_background(self, input: RemoveBackgroundResquest):
        image = decode_base64_to_image(input.image)
        output = rembg.remove(
            image,
            session=rembg.new_session(input.model),
            only_mask=False,
            alpha_matting=False,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        return {"image": encode_pil_to_base64(output), "model": input.model}

    def add_api_route(self, path: str, endpoint, **kwargs):
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(
                path, endpoint, dependencies=[Depends(self.auth)], **kwargs
            )
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in self.credentials:
            if compare_digest(
                credentials.password, self.credentials[credentials.username]
            ):
                return True

        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [str(title.lower()) for title in scripts.scripts_txt2img.titles]
        i2ilist = [str(title.lower()) for title in scripts.scripts_img2img.titles]

        return ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

    def init_default_script_args(self, script_runner):
        # find max idx from the scripts in runner and generate a none array to init script_args
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # None everywhere except position 0 to initialize script args
        script_args = [None] * last_arg_index
        script_args[0] = 0

        # get default values
        with gr.Blocks():  # will throw errors calling ui function without this
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from : script.args_to] = ui_default_values
        return script_args

    def init_script_args(
        self,
        request,
        default_script_args,
        selectable_scripts,
        selectable_idx,
        script_runner,
    ):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[
                selectable_scripts.args_from : selectable_scripts.args_to
            ] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts and (len(request.alwayson_scripts) > 0):
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script == None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"always on script {alwayson_script_name} not found",
                    )
                # Selectable script in always on script param check
                if alwayson_script.alwayson == False:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Cannot have a selectable script in the always on scripts params",
                    )
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    script_args[
                        alwayson_script.args_from : alwayson_script.args_to
                    ] = request.alwayson_scripts[alwayson_script_name]["args"]
        return script_args

    def text2imgapi(self, txt2imgreq: StableDiffusionTxt2ImgProcessingAPI):
        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui()
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(
                script_runner
            )
        selectable_scripts, selectable_script_idx = self.get_selectable_script(
            txt2imgreq.script_name, script_runner
        )

        populate = txt2imgreq.copy(
            update={  # Override __init__ params
                "sampler_name": validate_sampler_name(
                    txt2imgreq.sampler_name or txt2imgreq.sampler_index
                ),
                "do_not_save_samples": not txt2imgreq.save_images,
                "do_not_save_grid": not txt2imgreq.save_images,
            }
        )
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop("script_name", None)
        args.pop(
            "script_args", None
        )  # will refeed them to the pipeline directly after initializing them
        args.pop("alwayson_scripts", None)

        script_args = self.init_script_args(
            txt2imgreq,
            self.default_script_arg_txt2img,
            selectable_scripts,
            selectable_script_idx,
            script_runner,
        )

        send_images = args.pop("send_images", True)
        args.pop("save_images", None)

        with self.queue_lock:
            p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_txt2img_grids
            p.outpath_samples = opts.outdir_txt2img_samples

            shared.state.begin()
            if selectable_scripts != None:
                p.script_args = script_args
                processed = scripts.scripts_txt2img.run(
                    p, *p.script_args
                )  # Need to pass args as list here
            else:
                p.script_args = tuple(script_args)  # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end()

        b64images = (
            list(map(encode_pil_to_base64, processed.images)) if send_images else []
        )

        return TextToImageResponse(
            images=b64images, parameters=vars(txt2imgreq), info=processed.js()
        )

    def img2imgapi(self, img2imgreq: StableDiffusionImg2ImgProcessingAPI):
        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui()
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(
                script_runner
            )
        selectable_scripts, selectable_script_idx = self.get_selectable_script(
            img2imgreq.script_name, script_runner
        )

        populate = img2imgreq.copy(
            update={  # Override __init__ params
                "sampler_name": validate_sampler_name(
                    img2imgreq.sampler_name or img2imgreq.sampler_index
                ),
                "do_not_save_samples": not img2imgreq.save_images,
                "do_not_save_grid": not img2imgreq.save_images,
                "mask": mask,
            }
        )
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop(
            "include_init_images", None
        )  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop("script_name", None)
        args.pop(
            "script_args", None
        )  # will refeed them to the pipeline directly after initializing them
        args.pop("alwayson_scripts", None)

        script_args = self.init_script_args(
            img2imgreq,
            self.default_script_arg_img2img,
            selectable_scripts,
            selectable_script_idx,
            script_runner,
        )

        send_images = args.pop("send_images", True)
        args.pop("save_images", None)

        with self.queue_lock:
            p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
            p.init_images = [decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_img2img_grids
            p.outpath_samples = opts.outdir_img2img_samples

            shared.state.begin()
            if selectable_scripts != None:
                p.script_args = script_args
                processed = scripts.scripts_img2img.run(
                    p, *p.script_args
                )  # Need to pass args as list here
            else:
                p.script_args = tuple(script_args)  # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end()

        b64images = (
            list(map(encode_pil_to_base64, processed.images)) if send_images else []
        )

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return ImageToImageResponse(
            images=b64images, parameters=vars(img2imgreq), info=processed.js()
        )

    def extras_single_image_api(self, req: ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict["image"] = decode_base64_to_image(reqDict["image"])

        with self.queue_lock:
            result = postprocessing.run_extras(
                extras_mode=0,
                image_folder="",
                input_dir="",
                output_dir="",
                save_output=False,
                **reqDict,
            )

        return ExtrasSingleImageResponse(
            image=encode_pil_to_base64(result[0][0]), html_info=result[1]
        )

    def extras_batch_images_api(self, req: ExtrasBatchImagesRequest):
        reqDict = setUpscalers(req)

        def prepareFiles(file):
            file = decode_base64_to_file(file.data, file_path=file.name)
            file.orig_name = file.name
            return file

        reqDict["image_folder"] = list(map(prepareFiles, reqDict["imageList"]))
        reqDict.pop("imageList")

        with self.queue_lock:
            result = postprocessing.run_extras(
                extras_mode=1,
                image="",
                input_dir="",
                output_dir="",
                save_output=False,
                **reqDict,
            )

        return ExtrasBatchImagesResponse(
            images=list(map(encode_pil_to_base64, result[0])), html_info=result[1]
        )

    def pnginfoapi(self, req: PNGInfoRequest):
        if not req.image.strip():
            return PNGInfoResponse(info="")

        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        items = {**{"parameters": geninfo}, **items}

        return PNGInfoResponse(info=geninfo, items=items)

    async def progressapi(self, req: ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return ProgressResponse(
                progress=0,
                eta_relative=0,
                state=shared.state.dict(),
                textinfo=shared.state.textinfo,
            )

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += (
                1
                / shared.state.job_count
                * shared.state.sampling_step
                / shared.state.sampling_steps
            )

        time_since_start = time.time() - shared.state.time_start
        eta = time_since_start / progress
        eta_relative = eta - time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()
        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        return ProgressResponse(
            progress=progress,
            model=shared.opts.data["sd_model_checkpoint"],
            eta_relative=eta_relative,
            state=shared.state.dict(),
            current_image=current_image,
            textinfo=shared.state.textinfo,
        )

    def interrogateapi(self, interrogatereq: InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        img = decode_base64_to_image(image_b64)
        img = img.convert("RGB")

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")

        return InterrogateResponse(caption=processed)

    def interruptapi(self):
        shared.state.interrupt()

        return {}

    def unloadapi(self):
        unload_model_weights()

        return {}

    def reloadapi(self):
        reload_model_weights()

        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if metadata is not None:
                options.update(
                    {
                        key: shared.opts.data.get(
                            key, shared.opts.data_labels.get(key).default
                        )
                    }
                )
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def set_config(self, req: Dict[str, Any]):
        for k, v in req.items():
            shared.opts.set(k, v)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [
            {"name": sampler[0], "aliases": sampler[2], "options": sampler[3]}
            for sampler in sd_samplers.all_samplers
        ]

    def get_upscalers(self):
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    async def get_sd_models(self):
        return [
            {
                "title": x.title,
                "model_name": x.model_name,
                "hash": x.shorthash,
                "sha256": x.sha256,
                "filename": x.filename,
                "config": find_checkpoint_config_near_filename(x),
                "cached": x in checkpoints_loaded,
            }
            for x in checkpoints_list.values()
        ]

    def get_hypernetworks(self):
        return [
            {"name": name, "path": shared.hypernetworks[name]}
            for name in shared.hypernetworks
        ]

    def get_face_restorers(self):
        return [
            {"name": x.name(), "cmd_dir": getattr(x, "cmd_dir", None)}
            for x in shared.face_restorers
        ]

    def get_realesrgan_models(self):
        return [
            {"name": x.name, "path": x.data_path, "scale": x.scale}
            for x in get_realesrgan_models(None)
        ]

    def get_prompt_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append(
                {"name": style[0], "prompt": style[1], "negative_prompt": style[2]}
            )

        return styleList

    def get_embeddings(self):
        db = sd_hijack.model_hijack.embedding_db

        # Reload embeddings
        db.load_textual_inversion_embeddings()

        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        def convert_embeddings(embeddings):
            return {
                embedding.name: convert_embedding(embedding)
                for embedding in embeddings.values()
            }

        return {
            "loaded": convert_embeddings(db.word_embeddings),
            "skipped": convert_embeddings(db.skipped_embeddings),
        }

    def refresh_checkpoints(self):
        shared.refresh_checkpoints()

    def create_embedding(self, args: dict):
        try:
            shared.state.begin()
            filename = create_embedding(**args)  # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()  # reload embeddings so new one can be immediately used
            shared.state.end()
            return CreateResponse(
                info="create embedding filename: {filename}".format(filename=filename)
            )
        except AssertionError as e:
            shared.state.end()
            return TrainResponse(info="create embedding error: {error}".format(error=e))

    def create_hypernetwork(self, args: dict):
        try:
            shared.state.begin()
            filename = create_hypernetwork(**args)  # create empty embedding
            shared.state.end()
            return CreateResponse(
                info="create hypernetwork filename: {filename}".format(
                    filename=filename
                )
            )
        except AssertionError as e:
            shared.state.end()
            return TrainResponse(
                info="create hypernetwork error: {error}".format(error=e)
            )

    def preprocess(self, args: dict):
        try:
            shared.state.begin()
            preprocess(
                **args
            )  # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return PreprocessResponse(info="preprocess complete")
        except KeyError as e:
            shared.state.end()
            return PreprocessResponse(
                info="preprocess error: invalid token: {error}".format(error=e)
            )
        except AssertionError as e:
            shared.state.end()
            return PreprocessResponse(info="preprocess error: {error}".format(error=e))
        except FileNotFoundError as e:
            shared.state.end()
            return PreprocessResponse(info="preprocess error: {error}".format(error=e))

    def train_embedding(self, args: dict):
        try:
            shared.state.begin()
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ""
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                embedding, filename = train_embedding(
                    **args
                )  # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return TrainResponse(
                info="train embedding complete: filename: {filename} error: {error}".format(
                    filename=filename, error=error
                )
            )
        except AssertionError as msg:
            shared.state.end()
            return TrainResponse(info="train embedding error: {msg}".format(msg=msg))

    def train_hypernetwork(self, args: dict):
        try:
            shared.state.begin()
            shared.loaded_hypernetworks = []
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ""
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return TrainResponse(
                info="train embedding complete: filename: {filename} error: {error}".format(
                    filename=filename, error=error
                )
            )
        except AssertionError as msg:
            shared.state.end()
            return TrainResponse(
                info="train embedding error: {error}".format(error=error)
            )

    def get_memory(self):
        try:
            import os, psutil

            process = psutil.Process(os.getpid())
            res = (
                process.memory_info()
            )  # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = (
                100 * res.rss / process.memory_percent()
            )  # and total memory is calculated as actual value is not cross-platform safe
            ram = {"free": ram_total - res.rss, "used": res.rss, "total": ram_total}
        except Exception as err:
            ram = {"error": f"{err}"}
        try:
            import torch

            if torch.cuda.is_available():
                s = torch.cuda.mem_get_info()
                system = {"free": s[0], "used": s[1] - s[0], "total": s[1]}
                s = dict(torch.cuda.memory_stats(shared.device))
                allocated = {
                    "current": s["allocated_bytes.all.current"],
                    "peak": s["allocated_bytes.all.peak"],
                }
                reserved = {
                    "current": s["reserved_bytes.all.current"],
                    "peak": s["reserved_bytes.all.peak"],
                }
                active = {
                    "current": s["active_bytes.all.current"],
                    "peak": s["active_bytes.all.peak"],
                }
                inactive = {
                    "current": s["inactive_split_bytes.all.current"],
                    "peak": s["inactive_split_bytes.all.peak"],
                }
                warnings = {"retries": s["num_alloc_retries"], "oom": s["num_ooms"]}
                cuda = {
                    "system": system,
                    "active": active,
                    "allocated": allocated,
                    "reserved": reserved,
                    "inactive": inactive,
                    "events": warnings,
                }
            else:
                cuda = {"error": "unavailable"}
        except Exception as err:
            cuda = {"error": f"{err}"}
        return MemoryResponse(ram=ram, cuda=cuda)

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)
