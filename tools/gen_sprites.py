"""Genera gli sprite AI in locale con Z-Image-Turbo (GPU), toglie lo sfondo e li salva PNG 512x512.
uso: python tools/gen_sprites.py [N_CANDIDATI] [solo_id ...]"""
import os, sys, glob
import torch
from diffusers import ZImagePipeline
from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "sprites")
CAND = os.path.join(ROOT, "assets", "candidati")
MODEL = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Z-Image-Turbo-unsloth-bnb-4bit/snapshots/*"))[0]

STYLE = ("absurd derpy meme cartoon sticker, weird and stupid looking, single character centered, full body visible, thick black outline, "
         "bold flat colors, soft cel shading, mobile game asset, plain pure white background, no text")
POTATO = "a round chubby potato character with big googly eyes and a goofy face, "
ITEMS = {
    # livelli (si fondono)
    "p1": POTATO + "tiny baby potato crying huge cartoon tears, wearing a pacifier",
    "p2": POTATO + "potato with a ridiculous blond mullet haircut and sunglasses, smug grin",
    "p3": POTATO + "sweaty potato with hairy armpits, arms raised, green stink lines around it",
    "p4": POTATO + "influencer potato taking a selfie with a smartphone, duck lips, ring light",
    "p5": POTATO + "extremely muscular bodybuilder potato flexing huge biceps, tank top, veins",
    "p6": POTATO + "cosmic potato floating in space, galaxy swirls on its skin, dripping with butter, third eye",
    "p7": POTATO + "gigantic supreme emperor potato boss with golden crown, red royal cape, laser eyes, "
                   "glowing aura, evil laugh, epic",
    # oggetti senza senso (rimbalzano, non si fondono)
    "papera": "a yellow rubber duck wearing a tiny cowboy hat, surprised face",
    "ciabatta": "a single blue plastic flip-flop slipper with an angry face",
    "water": "a white toilet with googly eyes and its lid open like a screaming mouth",
    "banana": "a banana wearing sunglasses doing a dab dance move",
    "pesce": "a confused fish wearing a business tie, flopping",
    "cono": "an orange traffic cone with a monocle and a mustache",
    "galactique": POTATO + "ultimate galactic god potato made of a swirling purple and blue galaxy, glowing stars inside its body, "
                  "cosmic energy aura, tiny planets orbiting around it, epic and majestic, final boss",
    "patacoin": "a shiny thick gold coin with a cute cartoon potato face engraved in the center, sparkle, game currency icon, "
                "front view, perfectly round",
    # patate speciali assurde
    "samurai": POTATO + "samurai potato holding a katana sword above its head, red headband, angry battle cry, kimono",
    "soldat": POTATO + "french soldier potato with green army helmet, camouflage uniform, big mustache, saluting",
    "tomate": "a funny red tomato character as a ghost hunter, wearing a beige jumpsuit and a big proton pack backpack, "
              "holding a laser blaster, determined crazy eyes, full body",
    "fantome": "a cute silly white cartoon ghost with tongue out, wavy bottom, floating",
    "cosmo": POTATO + "astronaut potato inside a round glass space helmet, white space suit, floating, tiny stars and rings around",
}

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
only = sys.argv[2:] or list(ITEMS)
os.makedirs(OUT, exist_ok=True); os.makedirs(CAND, exist_ok=True)
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")


def cut_bg(img):
    """Sfondo bianco -> trasparente: solo il bianco collegato ai bordi (gli occhi bianchi restano)."""
    a = np.asarray(img.convert("RGB")).astype(int)
    white = (a.min(axis=2) > 225) & (a.max(axis=2) - a.min(axis=2) < 25)
    lab, _ = ndimage.label(white)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge[edge > 0])
    bg = ndimage.binary_opening(bg, iterations=1)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = np.asarray(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(1.2)))
    out = img.convert("RGBA"); out.putalpha(Image.fromarray(alpha)); return out

for key in only:
    for s in range(n):
        g = torch.Generator("cuda").manual_seed(1000 + s)
        img = pipe(prompt=f"{ITEMS[key]}. {STYLE}", height=1024, width=1024,
                   num_inference_steps=9, guidance_scale=0.0, generator=g).images[0]
        img.save(os.path.join(CAND, f"raw_{key}_{s}.png"))
        cut = cut_bg(img)
        cut = cut.crop(cut.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox())
        side = int(max(cut.size) * 1.04)
        sq = Image.new("RGBA", (side, side)); sq.paste(cut, ((side - cut.width) // 2, (side - cut.height) // 2))
        sq = sq.resize((512, 512), Image.LANCZOS)
        name = f"{key}.png" if n == 1 else f"{key}_{s}.png"
        sq.save(os.path.join(OUT if n == 1 else CAND, name))
        print("ok", name, flush=True)
