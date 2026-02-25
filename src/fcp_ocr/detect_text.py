"""
# Detect text in video (FCPXML)
(Once this module is implemented, maybe the whole 'fcp-silence-detector' should be upgraded to something else with multiple features including silence detection and text detection. Make the front-end your first)

## To make sure you don't place a marker every frame, set a time window parameter.
If the desired text is detected:
    Place '{text} start' and set state='detected'.
If state=='detected' and {text} is no longer detected for seconds=1:
    Place '{text} end' and set state='not detected'.

## Features
- Option to set a buffer range like buffer_seconds=0.5.

  If the pure detection gives
  ..........|--text---|..........,

  offer an option buffer_seconds=0.5 to place markers
  ......v.................v......

  because there could be other effects going on on screen before the desired text detectable clearly detectable by OCR.
- Should I use pyautogui OCR, or OpenCV image detection? Which one is more accurate? Which one is less expensive?
  Go with OCR first. This is a "Detect text" module.
- Allow multiple text detection maybe? Usually, my main use for this module is to remove the save or load screens in gaming videos. So the workflow is 'Open the menu' -> 'Click Save or Save As or Load or whatever' until the obvious text keyword "Saving..." appears on the screen. The pre-workflow can take more than the pre-determined time (like 0.5 seconds), I might just want to set multiple keywords for the detection condition and remove all of them by placing Markers.

## Optimization
- Sampling. Offer an option to perform OCR (expensive) at every n frames (or m milliseconds).
- Crop. Maybe first by declaring top-left and bottom-right pixels to scan. Add more user-friendly features later like declaring it with percentage.
"""

from tqdm import tqdm
import numpy as np
import pytesseract as tess
import cv2
import datetime
from joblib import Parallel, delayed

# Have OCR ready
tess.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# screen region to scan
def crop_frame(frame: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    assert top < bottom
    assert left < right

    output = frame[top:bottom, left:right]
    #print(f"Crop to top, left, bottom, right = {top, left, bottom, right}")

    return output

# OpenCV to go frame by frame

def detect_text_from_string(string: str, text: str):
    assert isinstance(text, str)
    assert text != ''

    x = string.find(text)
    if x != -1:
        # found the text in string
        return text
    return ''

def format_timedelta(td: datetime.timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    output = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return output

def seconds_to_frame(seconds: float, fps: int) -> int:
    assert seconds > -1
    assert fps > 0

    output = round(seconds * fps)
    return output

def frame_to_seconds(frames: int, fps: int) -> int:
    assert frames > -1
    assert fps > 0

    output = round(frames / fps)
    return output

def parse_target(string: str, w_max: int, h_max: int) -> list[int]:
    """
    "text:1,2,3,4" -> ['text', 1, 2, 3, 4]
    """
    assert h_max > 0
    assert w_max > 0

    text, numbers = string.rsplit(':', maxsplit=1)
    output = [text]
    a = [round(float(x.strip())) for x in numbers.split(',')]
    if a == [0, 0, 0, 0]:
        a = [0, 0, w_max, h_max]
    output += a

    return output

def detect_texts_from_video(file_path: str='', target: list[str]=[], skip_frames: int=1, skip_seconds: float=0.0, mode: str='and', debug: bool=False, optimize: bool=True):
    """
    target = ['abc:0,0,10,10', 'xyz:20,20,400,400', ...]
    mode = 'and' or 'or'
    returns the info as a list of dictionaries.
    [{'timestamp': 'hh:mm:ss', 'detected': 'abc xyz'}, {...}, ...]
    """
    assert file_path is not None
    assert isinstance(file_path, str)
    assert isinstance(target, list)
    assert skip_frames > 0
    assert mode in {'and', 'or'}
    for s in target:
        assert isinstance(s, str)
    print(f"DEBUG: target: {target}")

    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        raise RuntimeError("Could not open video file")

    # Determine cropping region
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    targets = [parse_target(string=t, w_max=width, h_max=height) for t in target]

    output = []

    # Scan every skip_frames frames
    fps = video.get(cv2.CAP_PROP_FPS)
    if skip_seconds != 0:
        skip_frames = seconds_to_frame(skip_seconds, fps)
    print(f"Checking every {skip_frames} frames")
    frame_count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count)):
        ret, frame = video.read()
        if not ret:
            break

        if (i % skip_frames) != 0:
            continue

        # frame is a NumPy array (H x W x 3)
        # OCR capture data to be added to the final output
        d = {'time': '', 'detected': ''}

        # Image processing to improve Tesseract OCR accuracy
        # Grayscale for OCR Scan
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # OCR capture per each target ['abc', 0, 0, 100, 100]
#       detected = []
        # OCR per each target text and region
#        for t in targets:
#            # If optimize==True, ignore all target scans as long as one of them is detected.
#            # only for mode=='or' of course.
#            if optimize and (mode=='or') and detected:
#                break
#
#            # Crop for OCR Scan
#            img = crop_frame(frame=frame, left=t[1], top=t[2], right=t[3], bottom=t[4])
#            #print(f"cropped image for {t}: {img}")
#
#            # Upscale for PyTesseract
#            scale = 2
#            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#
#            # OCR Scan
#            captured_string = tess.image_to_string(img)
#            if debug:
#                print(f"Captured string in searching for {t}: {captured_string}")
#
#            # Detect text
#            detected_text = detect_text_from_string(string=captured_string, text=t[0])
#            if not detected_text:
#                # Normalize
#                img2 = cv2.equalizeHist(img)
#                captured_string = tess.image_to_string(img2)
#                detected_text = detect_text_from_string(string=captured_string, text=t[0])
#            if not detected_text:
#                # Normalize
#                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#                img2 = clahe.apply(img)
#                captured_string = tess.image_to_string(img2)
#                detected_text = detect_text_from_string(string=captured_string, text=t[0])
#
#            # Something detected
#            if detected_text:
#                detected.append(detected_text)
#                if debug:
#                    print(f"Detected string: {detected_text}")
#            if debug:
#                cv2.imshow("debug frame", img)
#                cv2.waitKey(0)

        # get parallel
        # if this experiment is successful, further optimize here by giving a threshold for parallel computing ("go parallel for targets more than N")
        def ocr_on_roi(frame, t, debug):
            # Crop for OCR Scan
            img = crop_frame(frame=frame, left=t[1], top=t[2], right=t[3], bottom=t[4])
            #print(f"cropped image for {t}: {img}")

            # Upscale for PyTesseract
            scale = 2
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # OCR Scan
            captured_string = tess.image_to_string(img)
            if debug:
                print(f"Captured string in searching for {t}: {captured_string}")

            # Detect text
            detected_text = detect_text_from_string(string=captured_string, text=t[0])
            if not detected_text:
                # Normalize
                img2 = cv2.equalizeHist(img)
                captured_string = tess.image_to_string(img2)
                detected_text = detect_text_from_string(string=captured_string, text=t[0])
            if not detected_text:
                # Normalize
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img2 = clahe.apply(img)
                captured_string = tess.image_to_string(img2)
                detected_text = detect_text_from_string(string=captured_string, text=t[0])

            output = ''
            # Something detected
            if detected_text:
                #detected.append(detected_text)
                output = detected_text
                if debug:
                    print(f"Detected string: {detected_text}")
            if debug:
                cv2.imshow("debug frame", img)
                cv2.waitKey(0)

            return output

        parallel_obj = Parallel(n_jobs=-1)
        # list of detected texts ('' if none detected from target)
        detected_from_frame = parallel_obj(delayed(ocr_on_roi)(frame, t, debug) for t in targets)

        # if mode == 'and', make sure all texts were detected.
        if (mode == 'and') and (detected_from_frame != [t[0] for t in targets]):
            continue

        # Process the scanned data
        if not all(i == '' for i in detected_from_frame):
            d['detected'] = ' '.join(s for s in detected_from_frame if s)
            td = datetime.timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
            d['time'] = td.total_seconds()
            output.append(d)
            print(f"detected from the current frame: {d['detected']}")
            #print(f"OCR result so far: {output}")

    video.release()

    #print(f"OCR result before filter_match: {output}")
    #output = filter_match(targets=targets, detected=output, mode=mode)
    #print(f"OCR result after filter_match: {output}")

    return output
