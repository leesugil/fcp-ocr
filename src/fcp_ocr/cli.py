#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
import cv2

from . import detect_text
from . import place_markers
from fcp_io import fcpxml_io

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
    # fcp-ocr --targetp Save:48,45.5,52,48 --targetp Save:21,91,25,95 --targetp Load:21,91,25,95 --targetp 'Save As:48,52,52,55' --targetp Load:48,57.5,52,61 --targetp Saving:46,47,54,53 --targetp OBS:21,11,27,14 --targetp Stream:68,58,78,61 --targetp Record:68,60,78,64 --targetp Game:13,27,25,33 --targetp Tip:13,27,25,33 --targetp Game:45,25,55,29 --targetp Option:45,32,55,36 --targetp Photo:45,39,55,42 --targetp Mode:45,39,55,42 --targetp Option:46,1,55,5 --skip-seconds=1.5 --ocr_mode=or --affix=ocr_marked_ --optimize <filepath>
    parser = argparse.ArgumentParser(description="Detect texts in video (OCR), place FCP Markers")
    parser.add_argument("fcpxml_filepath", help="Absolute filepath to fcpxml (required)")
    parser.add_argument("--keyword", type=str, default='silence', help="Keyword to be used in Marker description")
    parser.add_argument("--event", action="store_true", help="Add this if the fcpxml file is exported from an Event item, not a Project in FCP.")

    # video/OCR related
    parser.add_argument("--target", action='append', type=str, help="texts to search in video. TEXT:left,top,right,bottom")
    parser.add_argument("--targetp", action='append', type=str, help="texts to search in video. TEXT:left%,top%,left%,right%,bottom%")
    parser.add_argument("--skip-frames", type=int, default=1, help="Perform OCR scanning on every X frames. Do not use with --skip-seconds.")
    parser.add_argument("--skip-seconds", type=float, default=0, help="Perform OCR scanning on every X seconds. Do not use with --skip-frames.")
    parser.add_argument("--ocr_mode", type=str, default='and', help="Whether the list of texts for the text detection is meant for AND conditions or OR conditions. 'and' or 'or' only.")

    # output
    parser.add_argument("--affix", type=str, default='ocr_marked_', help="affix to modify the output filename")

    # debug mode
    parser.add_argument("--debug", action='store_true', help="(experimental) display debug messages.")

    # optimized mode
    parser.add_argument("--optimize", action='store_true', help="(experimental) improves speed by detecting at most one text per target frame. useful when scanning multiple regions per frame but all you care is whether something was detected or not. 0 or 1")

    args = parser.parse_args()

    xf = fcpxml_io.clean_filepath(args.fcpxml_filepath)
    vf = fcpxml_io.clean_filepath(fcpxml_io.parse_fcpxml_filepath(xf))
    af = vf
    print(f"fcpxml file: {xf}")
    print(f"video file: {vf}")
    print(f"audio file: {af}")

     # detect text (OCR)
    target = args.target
    width, height = get_resolution(vf)
    if args.targetp:
        target = parse_targetp(args.targetp, width, height)
    detected_texts = detect_text.detect_texts_from_video(file_path=vf, target=target, skip_frames=args.skip-frames, skip_seconds=args.skip-seconds, mode=args.ocr_mode, debug=args.debug, optimize=args.optimize)

    tree, root = fcpxml_io.get_fcpxml(xf)
    fps = fcpxml_io.get_fps(root)
    place_markers.place(root=root, texts=detected_texts, fps=fps, keyword=args.keyword, in_event=args.event)

    fcpxml_io.save_with_affix(tree=tree, src_filepath=xf, affix=args.affix)

if __name__ == "__main__":
    main()
