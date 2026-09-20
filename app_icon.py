"""Original geometric nightscape mark; shared by Tk and native packaging.

No external artwork or fonts. Packaging assets are reproducible from this source.
"""
from pathlib import Path
from PIL import Image, ImageDraw


def icon_image(size=256):
    image = Image.new('RGBA', (1024, 1024))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((48, 48, 976, 976), radius=216, fill='#101e33')
    draw.rounded_rectangle((64, 64, 960, 960), radius=202, outline='#294560', width=12)
    # A rising arc and separate horizon suggest a wide night-sky field.
    draw.arc((170, 250, 870, 850), 38, 213, fill='#65d7dc', width=48)
    draw.arc((230, 315, 810, 797), 42, 170, fill='#367694', width=22)
    # The main star remains a distinct silhouette at 16 and 32 pixels.
    draw.polygon([(660, 192), (700, 310), (824, 350), (700, 390),
                  (660, 510), (620, 390), (496, 350), (620, 310)], fill='#edf9ff')
    draw.ellipse((284, 285, 330, 331), fill='#93b6da')
    draw.ellipse((408, 184, 434, 210), fill='#65d7dc')
    draw.line([(312, 740), (712, 740)], fill='#edf9ff', width=28)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def install_icon(root):
    from PIL import ImageTk
    # default=True also covers subsequently created child workspaces.
    root._nightscape_icons = [ImageTk.PhotoImage(icon_image(size), master=root)
                              for size in (16, 32, 48, 128, 256)]
    root.iconphoto(True, *root._nightscape_icons)


def build_icons(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    image = icon_image(1024)
    image.save(folder / 'nightscape.png')
    image.save(folder / 'nightscape.ico', sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
    image.save(folder / 'nightscape.icns')
    return folder


if __name__ == '__main__':
    build_icons(Path(__file__).resolve().parent / 'build' / 'app-icons')
