"""Assemble the new quantitative panel with the supplied vector composite."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / 'figures'
source = ROOT / 'manuscript/Fig1.pdf'
panel = FIG / 'Fig1i_cross_temporal_discriminability.pdf'
result = FIG / 'Fig1_with_discriminability_panel_i.pdf'
base = PdfReader(source)
replacement = PdfReader(panel)
assert len(base.pages) == len(replacement.pages) == 1
page = base.pages[0]
page.merge_transformed_page(replacement.pages[0], Transformation().translate(403, 25))
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({'/Title': 'Fig. 1 with cross-temporal discriminability in panel i',
                     '/Subject': 'Panel i replotted from source data as signed AUC-equivalent discriminability; other panels preserved.'})
with result.open('wb') as f:
    writer.write(f)

def render(path, dpi):
    doc = pdfium.PdfDocument(str(path))
    return doc[0].render(scale=dpi/72).to_pil().convert('RGB')

# Render both with the same engine and compare every pixel outside the replaced
# panel. This guards against accidental edits elsewhere in the composite.
dpi = 400
old_image, new_image = render(source, dpi), render(result, dpi)
old, new = np.asarray(old_image), np.asarray(new_image)
assert old.shape == new.shape
mask = np.ones(old.shape[:2], dtype=bool)
height_pt = float(PdfReader(source).pages[0].mediabox.height)
x0, x1 = int(403*dpi/72)-2, int(np.ceil(505*dpi/72))+2
y0, y1 = int((height_pt-157)*dpi/72)-2, int(np.ceil((height_pt-25)*dpi/72))+2
mask[y0:y1, x0:x1] = False
# The neighbouring panel's Hz label reaches x=406.68 pt. Preserve this
# neighbouring strip exactly below the panel-i header, including its colourbar.
mask[int((height_pt-134)*dpi/72):y1, x0:int(407*dpi/72)] = True
outside = np.any(old != new, axis=2) & mask
assert not outside.any(), f'{outside.sum()} pixels changed outside panel i'
png = FIG / 'Fig1_with_discriminability_panel_i.png'
new_image.save(png, dpi=(400, 400))
render(result, 600).save(FIG / 'Fig1_with_discriminability_panel_i.tiff', dpi=(600,600), compression='tiff_lzw')
render(result, 180).save(OUT / 'audit/Fig1_preview.png')
# Validate the generated figure directly; this distribution does not edit Word packages.
with Image.open(png) as check:
    check.verify()
panel_text = PdfReader(panel).pages[0].extract_text()
assert 'AUC' not in panel_text
assert 'discriminability' in panel_text and 'chance' in panel_text
report = json.loads((OUT / 'audit/figure_validation.json').read_text())
report.update({
    'source_figure_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'full_figure_size_mm': [float(page.mediabox.width)*25.4/72, float(page.mediabox.height)*25.4/72],
    'full_figure_png_pixels': list(new_image.size),
    'full_figure_png_dpi': dpi,
    'unmodified_panel_pixels_unchanged': True,
    'modified_pixel_region_pdf_points': [403, 25, 505, 157],
    'panel_svg_text_editable': '<text' in panel.with_suffix('.svg').read_text(),
    'panel_pdf_no_auc_label': True,
    'full_figure_assembly': 'Replacement panel merged over the original panel i region with a transparent notch preserving the adjacent panel f label and colourbar; original vectors outside panel i preserved.',
})
(OUT / 'audit/figure_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k.startswith(('full_', 'unmodified', 'panel_'))}, indent=2))
