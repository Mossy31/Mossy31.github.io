# Potato Chaos

Gioco "unisci le patate" senza senso, nel browser. Di [@Mossy31](https://instagram.com/cool_games_heel).

- `index.html`: tutto il gioco (fisica Matter.js in `vendor/`). Si pubblica così com'è su GitHub Pages.
- `assets/sprites/`: sprite generati in locale con Z-Image-Turbo (`tools/gen_sprites.py`, GPU).
- `assets/bg/`, `assets/ui/`: sfondi, icone dei boost e barattolo generati in locale (`tools/gen_ui.py`).
- `assets/voices/`: voci generate in locale con Kokoro, licenza Apache 2.0 (`tools/gen_voices.py`).
- Font: Lilita One (licenza OFL, `assets/fonts/OFL.txt`).
- `index.html?auto`: il gioco si gioca da solo (demo).

Provare in locale: `python3 -m http.server` nella cartella e aprire http://localhost:8000
