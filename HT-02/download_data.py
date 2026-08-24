"""
Construccion del dataset de sprites de Pokemon para la DCGAN.

Descarga los sprites front_default (96x96 PNG con canal alfa) desde el
repositorio publico PokeAPI/sprites en GitHub, los compone sobre fondo
solido para eliminar la transparencia, y los guarda como PNG RGB de 64x64.

IDs 1..898 = generaciones I a VIII (Bulbasaur .. Calyrex).

Uso:
    python download_pokemon_dataset.py --out data/pokemon --bg white
"""

import argparse
import os
import time
from io import BytesIO

import requests
from PIL import Image

BASE_URL = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/" "sprites/pokemon/{}.png"
)

BG_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def download_sprite(poke_id, session, retries=3, timeout=15):
    """Descarga un sprite. Devuelve bytes o None si no existe."""
    url = BASE_URL.format(poke_id)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None  # ese ID no tiene sprite front_default
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))  # backoff simple
    return None


def process_sprite(raw_bytes, size, bg_rgb):
    """PNG RGBA -> PIL Image RGB de size x size, sin transparencia."""
    img = Image.open(BytesIO(raw_bytes)).convert("RGBA")
    # El fondo transparente se vuelve negro si solo se hace .convert("RGB"),
    # lo cual introduce bordes duros. Componer explicitamente es mas limpio.
    background = Image.new("RGBA", img.size, bg_rgb + (255,))
    flat = Image.alpha_composite(background, img).convert("RGB")
    return flat.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="data/pokemon", help="carpeta destino de las imagenes"
    )
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument(
        "--end", type=int, default=898, help="ID final inclusivo (898 = hasta gen VIII)"
    )
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--bg", choices=list(BG_COLORS), default="white")
    ap.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="pausa entre requests, para no saturar GitHub",
    )
    args = ap.parse_args()

    # ImageFolder de torchvision espera imagenes dentro de una subcarpeta de
    # clase, aun cuando no haya clases reales. Por eso 'all/'.
    out_dir = os.path.join(args.out, "all")
    os.makedirs(out_dir, exist_ok=True)

    bg_rgb = BG_COLORS[args.bg]
    session = requests.Session()
    session.headers.update({"User-Agent": "dcgan-coursework/1.0"})

    saved, missing = 0, []
    total = args.end - args.start + 1

    for i in range(args.start, args.end + 1):
        dest = os.path.join(out_dir, f"{i:04d}.png")
        if os.path.exists(dest):  # reanudable
            saved += 1
            continue

        raw = download_sprite(i, session)
        if raw is None:
            missing.append(i)
        else:
            try:
                process_sprite(raw, args.size, bg_rgb).save(dest)
                saved += 1
            except Exception as e:
                print(f"  [warn] fallo al procesar {i}: {e}")
                missing.append(i)

        done = i - args.start + 1
        if done % 50 == 0 or done == total:
            print(f"{done}/{total} procesados | guardados: {saved}")
        time.sleep(args.delay)

    print(f"\nListo. {saved} imagenes en {out_dir}")
    if missing:
        print(f"Sin sprite ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
