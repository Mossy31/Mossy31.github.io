"""Genera in locale (Z-Image-Turbo) la grafica dell'interfaccia: icone dei boost, pareti del barattolo, nastro del pericolo.
uso: python tools/gen_ui.py [N_CANDIDATI] [solo_id ...]   -> assets/candidati/ui_<id>_<n>.png"""
import os, sys, glob
import numpy as np
import torch
from diffusers import ZImagePipeline
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "assets", "candidati")
MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Z-Image-Turbo-unsloth-bnb-4bit/snapshots/*"))[0]

ICON = ("mobile game icon, cartoon sticker, thick black outline, bold flat colors, soft cel shading, "
        "single object centered, plain pure white background, no text")
MATERIAL = ("made of glossy dark purple crystal with gold metal trim and gold rivets, glowing magenta edges, "
            "cartoon mobile game asset, thick black outline, plain pure white background, no text")
# id -> (prompt, larghezza, altezza, togli sfondo)
ITEMS = {
    "bomba":    ("a round black cartoon bomb with a lit sparkling fuse, " + ICON, 1024, 1024, True),
    "scossa":   ("a yellow lightning bolt with shaking motion lines, electric sparks, " + ICON, 1024, 1024, True),
    "calamita": ("a red horseshoe magnet with silver tips and magnetic wave lines, " + ICON, 1024, 1024, True),
    "reach":    ("two glowing cyan chain links connected by a crackling energy beam, " + ICON, 1024, 1024, True),
    "pugno":    ("a big red cartoon boxing glove punching forward, side view, impact motion lines, " + ICON, 1024, 1024, True),
    "spin":     ("a cyan circular rotating arrow forming a full loop, swirl motion, spin icon, " + ICON, 1024, 1024, True),
    "parete":   ("a single tall straight vertical pillar seen from the front, " + MATERIAL, 512, 1536, True),
    "fondo":    ("a single long straight horizontal beam seen from the front, " + MATERIAL, 1536, 512, True),
    "nastro":   ("a single long straight horizontal red and black hazard warning tape strip with diagonal stripes, "
                 "glowing red, cartoon game asset, thick black outline, plain pure white background, no text", 1536, 512, True),
    "overlord": ("a big round brown potato villain character, the potato skin and potato body clearly visible, "
                 "a grey iron half mask covering only the eyes and forehead with glowing green eye slits, "
                 "a huge evil grinning potato mouth visible under the mask, a flowing dark green cape with a high collar, "
                 "no hood, small grey metal gauntlets, green glowing energy in one hand, epic final boss, full body, " + ICON,
                 1024, 1024, True),
}
# sfondi: stesso stile per tutti, scuri e semplici, centro vuoto
BG = ("minimalist flat vector illustration, soft gradient shading, limited color palette, dark and slightly desaturated, "
      "simple shapes, large empty calm area in the center, vertical mobile game background, no characters, no text")
for k, desc in {
    "notte": "a quiet night sky over rolling potato fields, crescent moon, few stars",
    "cucina": "a cozy kitchen at night, shelves with jars, a pot on a stove",
    "spazio": "outer space with a few planets and a distant potato-shaped moon",
    "parigi": "paris rooftops at dusk with the eiffel tower in the distance",
    "fritte": "a volcano landscape made of french fries with ketchup rivers",
    "vapor": "a retro synthwave sunset with a grid floor and palm trees",
    "castello": "a haunted castle on a hill under a full moon, purple fog",
    "abisso": "deep underwater scene with seaweed, bubbles and light rays",
}.items():
    ITEMS["bg_" + k] = (f"{desc}. {BG}", 768, 1280, False)


def cut_bg(img):
    a = np.asarray(img.convert("RGB")).astype(int)
    white = (a.min(axis=2) > 225) & (a.max(axis=2) - a.min(axis=2) < 25)
    lab, _ = ndimage.label(white)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = ndimage.binary_opening(np.isin(lab, edge[edge > 0]), iterations=1)
    alpha = Image.fromarray(np.where(bg, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    out = img.convert("RGBA"); out.putalpha(alpha)
    return out.crop(out.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox())


n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
os.makedirs(CAND, exist_ok=True)
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
for key in (sys.argv[2:] or ITEMS):
    prompt, w, h, cut = ITEMS[key]
    for s in range(n):
        img = pipe(prompt=prompt, width=w, height=h, num_inference_steps=9, guidance_scale=0.0,
                   generator=torch.Generator("cuda").manual_seed(2000 + s)).images[0]
        (cut_bg(img) if cut else img).save(os.path.join(CAND, f"ui_{key}_{s}.png"))
        print("ok", key, s, flush=True)
