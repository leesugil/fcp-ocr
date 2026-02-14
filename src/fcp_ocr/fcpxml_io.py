import xml.etree.ElementTree as ET
import os
import shutil

def get_fcpxml(filepath):
    """
    returns tree, root
    """
    # structure filepath
    filepath = os.path.abspath(filepath)
    d, b = os.path.split(filepath)
    fcpxml_filename = 'Info.fcpxml'
    filepath = os.path.join(filepath, fcpxml_filename)

    # get the relevant xml tree from FCPXML
    tree = ET.parse(filepath)
    root = tree.getroot()

    return tree, root

def get_clip(root, sync=False):
    if sync:
        return root.find(".//sync-clip[@format='r1']")
    else:
        return root.find(".//asset-clip")

def get_offset(clip, sync=False):
    offset = 0.0
    if sync:
        asset_clip1 = clip.find(".//asset-clip[@ref='r2']") # this is the source video
        asset_clip2 = asset_clip1.find(".//asset-clip[@ref='r3']") # this is the added audio "track"
        offset = asset_clip2.get('offset')
        offset = eval(offset.rstrip('s')) # as a float
    return offset

def save(tree, filepath, affix=''):
    # Save the new FCPXML file
    filepath = os.path.abspath(filepath)
    d, b = os.path.split(filepath)
    fcpxml_filename = 'Info.fcpxml'
    filepath = os.path.join(filepath, fcpxml_filename)

    src_filepath = os.path.join(d, b) # still *.fcpxmld folder path
    new_filepath = os.path.join(d, affix+b) # silence_marked_*.fcpxmld a new foler path
    shutil.copytree(src_filepath, new_filepath, dirs_exist_ok=True) # create the new folder
    destination_filepath = os.path.join(new_filepath, fcpxml_filename) # *.fcpxml file
    ET.indent(tree, space="\t", level=0)
    tree.write(destination_filepath, encoding='UTF-8', xml_declaration=True)

