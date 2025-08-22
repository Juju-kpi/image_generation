from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from diffusers import DiffusionPipeline
import torch
import uuid
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model at startup
model_name = "Qwen/Qwen-Image"
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
}

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def generate_image(request: Request, prompt: str = Form(...), aspect: str = Form("16:9")):
    width, height = aspect_ratios.get(aspect, (1664, 928))
    
    full_prompt = prompt + positive_magic["en"]
    image = pipe(
        prompt=full_prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device).manual_seed(42)
    ).images[0]

    filename = f"static/generated_{uuid.uuid4().hex}.png"
    image.save(filename)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_path": "/" + filename
    })

@app.get("/download/{filename}", response_class=FileResponse)
async def download_file(filename: str):
    return FileResponse(path=f"static/{filename}", filename=filename, media_type='image/png')
