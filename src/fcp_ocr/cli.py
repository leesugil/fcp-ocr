#!/usr/bin/env python3

import os
import argparse
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote
import cv2

from . import detect_text
from . import place_markers

def clean_filepath(line):
    output = os.path.abspath(line.strip())
    return output

def parse_fcpxml_filepath(xf):
    fcpxml_filename = 'Info.fcpxml'
    fcpxml_filepath = os.path.join(xf, fcpxml_filename)
    tree = ET.parse(fcpxml_filepath)
    root = tree.getroot()
    media_rep = root.find(".//media-rep[@kind='original-media']")
    output = media_rep.get('src')
    output = urlparse(output)
    output = unquote(output.path)
    return output

def parse_fcpxml_filepath_sync(xf):
    fcpxml_filename = 'Info.fcpxml'
    fcpxml_filepath = os.path.join(xf, fcpxml_filename)

    tree = ET.parse(fcpxml_filepath)
    root = tree.getroot()

    # video
    asset1 = root.find(".//asset[@id='r2']")
    media_rep1 = asset1.find(".//media-rep[@kind='original-media']")
    output1 = media_rep1.get('src')
    output1 = urlparse(output1)
    output1 = unquote(output1.path) # video

    # audio
    asset2 = root.find(".//asset[@id='r3']")
    media_rep2 = asset2.find(".//media-rep[@kind='original-media']")
    output2 = media_rep2.get('src')
    output2 = urlparse(output2)
    output2 = unquote(output2.path) # audio

    return output1, output2

def get_resolution(filepath):
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        raise RuntimeError("Could not open video file")
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    return (width, height)

def parse_targetp(targetp, width, height):
    """
    ['ABC XYZ:10,20,30,40', ...], 2000, 1000 -> ['ABC XYZ:100,400,300,800', ...]
    """
    output = []
    for t in targetp:
        target_list = detect_text.parse_target(t, w_max=width, h_max=height)
        # width
        target_list[1] *= (width / 100)
        target_list[3] *= (width / 100)
        # height
        target_list[2] *= (height / 100)
        target_list[4] *= (height / 100)
        target = f"{target_list[0]}:{target_list[1]},{target_list[2]},{target_list[3]},{target_list[4]}"
        output.append(target)

    return output

def main():

    # Define possible arguments
    # ex)
    # fcp-ocr --targetp Save:48,46,51,48 --targetp Save:21,91,25,93 --targetp Load:21,91,25,93 --targetp 'Save As:48,52,52,55' --targetp Load:48,59,52,61 --targetp Saving:46,47,54,53 --targetp OBS:21,11,27,14 --targetp Stream:68,58,78,61 --targetp Record:68,62,78,64 --skip_seconds=1.0 --ocr_mode=or --affix=clahe_ocr_marked_ --debug=0 --sync=0 <filepath>
    parser = argparse.ArgumentParser(description="Detect texts in video (OCR), place FCP Markers")
    parser.add_argument("fcpxml_filepath", help="Absolute filepath to fcpxml (required)")

    # video/OCR related
    parser.add_argument("--target", action='append', type=str, help="texts to search in video. TEXT:left,top,right,bottom")
    parser.add_argument("--targetp", action='append', type=str, help="texts to search in video. TEXT:left%,top%,left%,right%,bottom%")
    parser.add_argument("--skip_frames", type=int, default=1, help="Perform OCR scanning on every X frames. Do not use with --skip_seconds.")
    parser.add_argument("--skip_seconds", type=float, default=0, help="Perform OCR scanning on every X seconds. Do not use with --skip_frames.")
    parser.add_argument("--ocr_mode", type=str, default='and', help="Whether the list of texts for the text detection is meant for AND conditions or OR conditions. 'and' or 'or' only.")

    # output
    parser.add_argument("--affix", type=str, default='ocr_marked_', help="affix to modify the output filename")

    # synched clip
    parser.add_argument("--sync", type=int, default=0, help="(experimental) synched clip. 0 or 1")

    # debug mode
    parser.add_argument("--debug", type=int, default=0, help="(experimental) display debug messages. 0 or 1")

    # optimized mode
    parser.add_argument("--optimize", type=int, default=1, help="(experimental) improves speed by detecting at most one text per target frame. useful when scanning multiple regions per frame but all you care is whether something was detected or not. 0 or 1")

    args = parser.parse_args()
    sync = True if args.sync == 1 else False
    debug = True if args.debug == 1 else False
    optimize = True if args.optimize == 1 else False

    xf = clean_filepath(args.fcpxml_filepath)
    vf = clean_filepath(parse_fcpxml_filepath(xf))
    af = vf
    if args.sync == 1:
        vf, af = parse_fcpxml_filepath_sync(xf)
        vf = clean_filepath(vf)
        af = clean_filepath(af)
    print(f"fcpxml file: {xf}")
    print(f"video file: {vf}")
    print(f"audio file: {af}")

     # detect text (OCR)
    target = args.target
    width, height = get_resolution(vf)
    if args.targetp:
        target = parse_targetp(args.targetp, width, height)
    detected_texts = detect_text.detect_texts_from_video(file_path=vf, target=target, skip_frames=args.skip_frames, skip_seconds=args.skip_seconds, mode=args.ocr_mode, debug=debug, optimize=optimize)

    place_markers.place(filepath=xf, texts=detected_texts, affix=args.affix, sync=sync)

if __name__ == "__main__":
    main()
