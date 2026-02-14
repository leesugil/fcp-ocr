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

#def detect_texts_from_string(string: str, texts: list[str]):
#    output = []
#
#    for t in texts:
#        d = detect_text_from_string(string, t)
#        if d != '':
#            output.append(d)
#
#    return output

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

#def filter_match(texts, detected, mode):
#    """
#    texts = 'abc xyz ab cd xy zw ...'
#    mode = 'and' or 'or'
#    """
#    output = []
#
#    # OR
#    if mode == 'or':
#        output = detected
#        return output
#
#    # AND
#    for d in detected:
#        if set(texts) == set(d['detected']):
#            output.append(d)
#
#    return output

#def detect_texts_from_video(file_path: str='', texts: list[str]=[], top: int=0, left: int=0, bottom: int=0, right: int=0, skip_frames: int=1, skip_seconds: int=0, mode='and'):
#    """
#    mode = 'and' or 'or'
#    returns the info as a list of dictionaries.
#    [{'timestamp': 'hh:mm:ss', 'detected': [a, b, c]}, {...}, ...]
#
#    detected_texts = detect_text.detect_texts_from_video(file_path=vf, target=args.target, skip_frames=args.skip_frames, skip_seconds=args.skip_seconds, mode=args.ocr_mode)
#    """
#    assert file_path is not None
#    assert isinstance(file_path, str)
#    assert len(texts) > 0 #if args.texts = '', then texts = [], raising assertion.
#    assert skip_frames > 0
#    assert mode in {'and', 'or'}
#    for s in texts:
#        assert isinstance(s, str)
#        assert s != ''
#    print(f"DEBUG: texts: {texts}")
#
#    video = cv2.VideoCapture(file_path)
#    if not video.isOpened():
#        raise RuntimeError("Could not open video file")
#
#    # Determine cropping region
#    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#    if (top, left, bottom, right) == (0, 0, 0, 0):
#        bottom = height
#        right = width
#    #print(f"top,left,bottom,right {top},{left},{bottom},{right}")
#
#    output = []
#
#    # Scan every skip_frames frames
#    fps = video.get(cv2.CAP_PROP_FPS)
#    if skip_seconds != 0:
#        skip_frames = seconds_to_frame(skip_seconds, fps)
#    frame_count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
#
#    for i in tqdm(range(frame_count)):
#        if (i % skip_frames) != 0:
#            continue
#
#        ret, frame = video.read()
#        if not ret:
#            break
#
#        # frame is a NumPy array (H x W x 3)
#        # Do something with frame
#        d = {'time': '', 'detected': []}
#
#        # Crop for OCR Scan
#        img = crop_frame(frame=frame, top=top, left=left, bottom=bottom, right=right)
#        #print(f"top,left,bottom,right {top},{left},{bottom},{right}")
#
#        # Grayscale for OCR Scan
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#        # Upscale for PyTesseract
#        #scale = 2
#        #img = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
#
#        # OCR Scan
#        captured_string = tess.image_to_string(img)
#        #print(f"Captured string: {captured_string}")
#
#        d['detected'] = detect_texts_from_string(string=captured_string, texts=texts)
#        #print(f"Detected string: {d['detected']}")
#        #cv2.imshow("debug frame", img)
#        #cv2.waitKey(1)
#
#        # Process the scanned data
#        if d['detected']:
#            td = datetime.timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
#            d['time'] = format_timedelta(td)
#            output.append(d)
#            print(f"detected from the current frame: {d['detected']}")
#            print(f"OCR result so far: {output}")
#
#    video.release()
#
#    print(f"OCR result before filter_match: {output}")
#    output = filter_match(texts=texts, detected=output, mode=mode)
#    print(f"OCR result after filter_match: {output}")
#
#    return output

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

def detect_texts_from_video(file_path: str='', target: list[str]=[], skip_frames: int=1, skip_seconds: float=0.0, mode: str='and', debug: bool=False):
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
        detected = []
        for t in targets:
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
            if detected_text:
                detected.append(detected_text)
                if debug:
                    print(f"Detected string: {detected_text}")
            if debug:
                cv2.imshow("debug frame", img)
                cv2.waitKey(0)

        if (mode == 'and') and (detected != [t[0] for t in targets]):
            continue

        # Process the scanned data
        if detected:
            d['detected'] = ' '.join(detected)
            td = datetime.timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
            #d['time'] = format_timedelta(td)
            d['time'] = td.total_seconds()
            output.append(d)
            print(f"detected from the current frame: {d['detected']}")
            #print(f"OCR result so far: {output}")

    video.release()

    #print(f"OCR result before filter_match: {output}")
    #output = filter_match(targets=targets, detected=output, mode=mode)
    #print(f"OCR result after filter_match: {output}")

    return output
