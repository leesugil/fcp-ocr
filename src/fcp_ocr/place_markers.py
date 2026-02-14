import xml.etree.ElementTree as ET
from tqdm import tqdm
from . import fcpxml_io

def place(filepath: str, texts: list[dict], affix: str, sync=False):
    """
    texts: [{'time': xx.xxx, 'detected': 'ABC XYZ ...'}, {...}, ...]
    """
    tree, root = fcpxml_io.get_fcpxml(filepath)
    asset_clip = fcpxml_io.get_clip(root, sync)
    offset = fcpxml_io.get_offset(asset_clip, sync)

    # Place OCR Markers
    for i, s in tqdm(enumerate(texts, start=1)):
        start_marker = ET.SubElement(asset_clip, "marker")
        start_marker.set("start", f"{s['time']+offset}s")
        start_marker.set("value", f"OCR detected {i}: {s['detected']}")
        start_marker.set("duration", "100/6000s")
        start_marker.set("completed", "0")

    fcpxml_io.save(tree, filepath, affix)

