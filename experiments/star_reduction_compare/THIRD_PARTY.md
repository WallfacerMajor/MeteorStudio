# Attribution and licensing

This isolated experimental prototype (engine.py, app.py and its tests) is
distributed under GPL-3.0-or-later. See COPYING.

The Siril PixelMath transfer expression is adapted from
DSA-Star_Reduction.py, copyright Rich Stevenson / Deep Space Astro,
GPL-3.0-or-later:
https://gitlab.com/free-astro/siril-scripts/-/blob/main/processing/DSA-Star_Reduction.py

Siril and StarNet are external, user-installed programs. Their executables and
model weights are not included or redistributed by this prototype. StarNet v2
is proprietary freeware, not an open-source component of this application.

The comparison uses the same StarNet-generated background in both methods.
The local RGB-linked screen-layer compression is an experimental variant, not
a reproduction of RC-Astro StarShrink, and is not claimed to outperform Siril.

Packaged Python dependencies retain their own licenses in the distribution.
Python: PSF license; NumPy: BSD; Pillow: HPND; tifffile: BSD-3-Clause;
OpenCV: Apache-2.0. Source is supplied alongside the standalone build.
