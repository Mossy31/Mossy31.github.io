"""Genera le voci del gioco IN LOCALE con Kokoro (licenza Apache 2.0) + effetti ffmpeg come nei video
(livello 1 = squittio, sempre più grave, boss = voce dell'oltretomba con bassi ed eco) -> assets/voices/*.mp3
uso: python tools/gen_voices.py [solo_id ...]"""
import os, sys, subprocess
import soundfile as sf
from kokoro import KPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "voices")
TMP = os.path.join(ROOT, "assets", "voices_raw")

S = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.02"
TRIM = f"{S},areverse,{S},areverse"          # niente silenzio prima/dopo: il nome parte subito
# id -> (frase, voce, velocità, effetti)
# Tutto in francese, una sola voce (ff_siwis): il tono scende a ogni livello, il boss è il più grave.
V = "ff_siwis"
LINES = {
    "p1": ("La patatine pleurnicheuse !", V, 1.30, "rubberband=pitch=1.80"),
    "p2": ("La patate coupe mulet !", V, 1.20, "rubberband=pitch=1.45"),
    "p3": ("La patate aux aisselles !", V, 1.10, "rubberband=pitch=1.15"),
    "p4": ("La patate influenceuse !", V, 1.00, "rubberband=pitch=0.92,bass=g=3:f=120:w=0.8"),
    "p5": ("La patate bodybuildée !", V, 0.95, "rubberband=pitch=0.78:tempo=0.95,bass=g=6:f=110:w=0.8"),
    "p6": ("La patate cosmique beurrée !", V, 0.90, "rubberband=pitch=0.66:tempo=0.92,bass=g=8:f=100:w=0.8,aecho=0.8:0.5:60:0.15"),
    "p7": ("La cosmo patate !", V, 0.85, "rubberband=pitch=0.6:tempo=0.9,bass=g=10:f=95:w=0.8,aecho=0.8:0.6:120|240:0.35|0.2"),
    "p8": ("La patate galactique !", V, 0.8,
           "rubberband=pitch=0.5:tempo=0.85,bass=g=12:f=90:w=0.8,aecho=0.8:0.55:70|140:0.3|0.18"),
    "overlord": ("Je suis l'overlord des patates !", V, 0.75,
                 "rubberband=pitch=0.45:tempo=0.85,bass=g=14:f=80:w=0.8,aecho=0.8:0.6:90|180|300:0.4|0.25|0.15"),
    "papera":   ("Coin coin ! Yihaa !", V, 1.15, "rubberband=pitch=1.6"),
    "ciabatta": ("La tong furieuse !", V, 1.15, "rubberband=pitch=1.0"),
    "water":    ("Aaaah ! Je suis des toilettes !", V, 1.1, "rubberband=pitch=0.9"),
    "banana":   ("Banane dab !", V, 1.2, "rubberband=pitch=1.3"),
    "pesce":    ("Bonjour, je suis le poisson manager.", V, 1.0, "rubberband=pitch=0.75"),
    "cono":     ("Cône distingué. Très distingué.", V, 0.9, "rubberband=pitch=0.7"),
    "samurai":  ("La patate samouraï ! Tchac !", V, 1.1, "rubberband=pitch=0.85,bass=g=3:f=120:w=0.8"),
    "soldat":   ("La patate soldat ! À l'attaque !", V, 1.15, "rubberband=pitch=0.8"),
    "tomate":   ("La tomate chasseuse de fantômes !", V, 1.15, "rubberband=pitch=1.25"),
    "fantome":  ("Bouuuuh !", V, 0.8, "rubberband=pitch=0.7,aecho=0.8:0.7:200|400:0.4|0.25"),
    "cosmo":    ("La cosmo patate !", V, 0.95, "rubberband=pitch=1.1,aecho=0.8:0.6:120|240:0.35|0.2"),
}

pipe = KPipeline(lang_code="f")
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
for key in (sys.argv[1:] or LINES):
    text, voice, speed, fx = LINES[key]
    audio = [r.audio.numpy() for r in pipe(text, voice=voice, speed=speed)]
    raw = os.path.join(TMP, f"{key}.wav")
    import numpy as np
    sf.write(raw, np.concatenate(audio), 24000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af",
                    f"{TRIM},{fx},acompressor=threshold=-18dB:ratio=4:attack=5:release=120,loudnorm=I=-15:TP=-1.5:LRA=7",
                    "-ar", "44100", "-ac", "1", "-b:a", "96k", os.path.join(OUT, f"{key}.mp3")], check=True)
    d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        os.path.join(OUT, f"{key}.mp3")], capture_output=True, text=True).stdout.strip()
    print(f"ok {key}: {d}s", flush=True)
