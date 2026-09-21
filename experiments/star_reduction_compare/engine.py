"""Isolated star-reduction experiment. Inputs and global Siril settings stay read-only."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import tifffile
from PIL import Image


def read_rgb(path):
    path = Path(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tf:
            try:
                pixels = tf.asarray()
            except (ValueError, NotImplementedError):
                import cv2
                pixels = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
                if pixels is None:
                    raise ValueError('无法解码此 TIFF')
                if pixels.ndim == 3:
                    pixels = pixels[..., :3][..., ::-1]
            tag = tf.pages[0].tags.get(34675)
            profile = tag.value if tag else None
        if pixels.ndim == 3 and pixels.shape[0] in (3, 4) and pixels.shape[-1] not in (3, 4):
            pixels = pixels.transpose(1, 2, 0)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[..., None], 3, axis=2)
        pixels = pixels[..., :3]
        maximum = np.iinfo(pixels.dtype).max if pixels.dtype.kind in 'ui' else 1
        return np.clip(pixels.astype(np.float32)/maximum, 0, 1), profile
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), np.float32)/255, im.info.get('icc_profile')


def write_rgb(path, array, profile=None):
    tags = [(34675, 'B', len(profile), profile, False)] if profile else []
    tifffile.imwrite(path, np.rint(np.clip(array, 0, 1)*65535).astype(np.uint16),
                     photometric='rgb', metadata=None, extratags=tags)


def midtones(x, m):
    return ((1-m)*x)/np.maximum(1e-12, m-(2*m-1)*x)


def reduce_local(original, starless, amount):
    """Screen-residual tone compression with one linked RGB gain per pixel.

    This experimental variant is deliberately distinct from DSA's channel-wise
    original/background transfer. It preserves the hue ratios of the extracted
    screen layer and bounds output between estimated background and original.
    """
    amount = float(np.clip(amount, 0, 1))
    if amount == 0:
        return original.copy()
    background = np.minimum(np.clip(starless, 0, 1), original)
    stars = np.clip((original-background)/np.maximum(1-background, 1e-6), 0, 1)
    peak = stars.max(axis=2)
    m = .5+.45*amount
    compressed = midtones(peak, m)
    gain = np.divide(compressed, peak, out=np.ones_like(peak), where=peak>1e-8)
    return np.clip(background+(1-background)*stars*gain[..., None], background, original)


class Comparison:
    def __init__(self, source, directory, siril=None, config=None):
        self.source = Path(source).resolve()
        self.directory = Path(directory).resolve()
        if self.directory == self.source.parent:
            raise ValueError('请选择独立输出目录')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.siril = Path(siril or r'C:\Program Files\Siril\bin\siril-cli.exe')
        default_config = Path(os.environ.get('LOCALAPPDATA', ''))/'siril/config.1.4.ini'
        self.config = Path(config or default_config)
        self.cancelled = False
        self.process = None

    def run_siril(self, name, commands, progress=lambda value: None):
        if not self.siril.is_file():
            raise FileNotFoundError('未找到 Siril，请选择 siril-cli 可执行文件')
        if not self.config.is_file():
            raise FileNotFoundError('未找到 Siril 配置，请选择已配置 StarNet 的配置文件')
        local_config = self.directory/'siril-local.ini'
        if not local_config.exists():
            shutil.copy2(self.config, local_config)
        script = self.directory/f'{name}.ssf'
        script.write_text('requires 1.2.0\nsetcpu 8\nset32bits\nsetext fits\n'+commands+'\nclose\n', encoding='utf-8')
        args = [str(self.siril), '-o', '-i', str(local_config), '-d', str(self.directory), '-s', str(script)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (self.directory/f'{name}.log').open('w', encoding='utf-8') as log:
            self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            creationflags=flags)
            for raw in iter(self.process.stdout.readline, b''):
                line = raw.decode('utf-8', errors='replace').strip()
                log.write(line+'\n'); log.flush()
                if line:
                    progress(line)
                if self.cancelled:
                    self.process.terminate()
            code = self.process.wait()
            self.process = None
        if self.cancelled:
            raise RuntimeError('已取消')
        if code:
            raise RuntimeError(f'Siril 处理失败，请查看 {name}.log')

    def prepare(self, progress=lambda value: None):
        progress('读取原图')
        pixels, profile = read_rgb(self.source)
        with self.source.open('rb') as source_file:
            fingerprint = hashlib.file_digest(source_file, 'sha256').hexdigest()
        meta_path = self.directory/'input.json'
        if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8'))['sha256'] != fingerprint:
            raise ValueError('此输出目录属于另一张照片，请创建新的输出目录')
        write_rgb(self.directory/'input.tif', pixels, profile)
        meta_path.write_text(json.dumps({'source': str(self.source), 'sha256': fingerprint,
            'shape': list(pixels.shape), 'nonlinear_input': True}, ensure_ascii=False, indent=2),encoding='utf-8')
        if not (self.directory/'starless.tif').exists():
            self.run_siril('separate', 'load input.tif\nsave original\nstarnet -nostarmask\nsave starless\nsavetif starless',progress)
        if not (self.directory/'starless.tif').exists():
            raise RuntimeError('Siril 未生成无星图，请检查 separate.log')
        return pixels, read_rgb(self.directory/'starless.tif')[0], profile

    def siril_reduce(self, amount, profile=None, progress=lambda value: None):
        # DSA-Star_Reduction transfer expression, executed by real Siril PixelMath.
        # The original script is GPL-3.0-or-later; see THIRD_PARTY.md.
        s = .5-.45*float(np.clip(amount, 0, 1))
        expression = f'~((~mtf(~{s:.8f},$original$)/~mtf(~{s:.8f},$starless$))*~$starless$)'
        self.run_siril('siril-reduce', f'load original.fits\npm "{expression}"\nsavetif siril-result',progress)
        result = read_rgb(self.directory/'siril-result.tif')[0]
        write_rgb(self.directory/'siril-result.tif', result, profile)
        return result

    def compare(self, amount=.55, progress=lambda value: None):
        original, starless, profile = self.prepare(progress)
        progress('计算本地方案')
        local = np.empty_like(original)
        for y in range(0, len(original), 256):
            local[y:y+256] = reduce_local(original[y:y+256],starless[y:y+256],amount)
        write_rgb(self.directory/'local-result.tif',local,profile)
        progress('调用 Siril 缩星')
        reference = self.siril_reduce(amount,profile,progress)
        metadata = {'amount': amount, 'siril_value': .5-.45*amount,
            'local_method': 'RGB-linked screen-layer midtone compression',
            'siril_method': 'DSA transfer expression executed in Siril PixelMath',
            'shared_starless': True, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        (self.directory/'comparison.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return original,local,reference,profile


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('output')
    parser.add_argument('--amount',type=float,default=.55)
    args=parser.parse_args()
    Comparison(args.source,args.output).compare(args.amount,lambda s:print(s,flush=True))
