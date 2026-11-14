"""Genera gli sfondi sbloccabili in locale con Z-Image-Turbo -> assets/bg/*.webp (verticali 600x1000)
uso: python tools/gen_backgrounds.py [solo_id ...]"""
import os, sys, glob
import torch
from diffusers import ZImagePipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "bg")
MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Z-Image-Turbo-unsloth-bnb-4bit/snapshots/*"))[0]
STYLE = ("vertical mobile game background, vibrant cartoon illustration, thick outlines, empty center area, "
         "no characters, no people, no text, no letters")
BGS = {
    "cuisine": "an absurd grandma kitchen with giant pots of boiling oil, hanging sausages, checkered floor, steam",
    "espace": "deep outer space with purple nebula, a planet shaped like a potato, stars, floating french fries",
    "paris": "paris street at sunset with the eiffel tower made of baguettes, giant croissants in the sky",
    "frites": "hell made of french fries, rivers of ketchup and mayonnaise lava, dramatic red sky",
    "vaporwave": "vaporwave retro sunset, neon pink grid floor, palm trees, greek statue head of a potato",
    "fantomes": "haunted castle at night, full moon, purple fog, spooky trees, cute cartoon style",
}
os.makedirs(OUT, exist_ok=True)
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
for key in (sys.argv[1:] or BGS):
    img = pipe(prompt=f"{BGS[key]}. {STYLE}", height=1280, width=768, num_inference_steps=9,
               guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(7)).images[0]
    img.resize((600, 1000)).save(os.path.join(OUT, f"{key}.webp"), quality=82, method=6)
    print("ok", key, flush=True)
