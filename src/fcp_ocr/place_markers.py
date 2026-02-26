import xml.etree.ElementTree as ET
from tqdm import tqdm
from fcp_io import fcpxml_io
from fcp_math import arithmetic

def place(root, texts: list[dict], fps: str, keyword: str, in_event: bool=False):
    """
    texts: [{'time': xx.xxx, 'detected': 'ABC XYZ ...'}, {...}, ...]
    """
    asset_clip = fcpxml_io.get_event_asset_clip(root) if in_event else fcpxml_io.get_spine_asset_clip(root)

    # Place OCR Markers
    for i, s in tqdm(enumerate(texts, start=1)):
        start = arithmetic.float2fcpsec(s['time'], fps)
        start_marker = ET.SubElement(asset_clip, "marker")
        start_marker.set("start", start)
        start_marker.set("value", f"{keyword} detected {i}: {s['detected']}")
        start_marker.set("duration", fps)
        start_marker.set("completed", "0")

