
from fcp_ocr import detect_text

def test_crop_frame():
    cases = [
                {
                'args': {
                    #frame: np.ndarray, left: int, top: int, right: int, bottom: int
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.crop_frame(case['args']) == case['expected']

def test_detect_text_from_string():
    cases = [
                {
                'args': {
                    #string: str, text: str
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.detect_text_from_string(case['args']) == case['expected']

def test_format_timedelta():
    cases = [
                {
                'args': {
                    #td: datetime.timedelta
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.format_timedelta(case['args']) == case['expected']

def test_seconds_to_frame():
    cases = [
                {
                'args': {
                    #seconds: float, fps: int
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.seconds_to_frame(case['args']) == case['expected']

def test_frame_to_seconds():
    cases = [
                {
                'args': {
                    #frames: int, fps: int
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.frame_to_seconds(case['args']) == case['expected']

def test_parse_target():
    cases = [
                {
                'args': {
                    #string: str, w_max: int, h_max: int
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.parse_target(case['args']) == case['expected']

def test_detect_texts_from_video():
    cases = [
                {
                'args': {
                    #file_path: str='', target: list[str]=[], skip_frames: int=1, skip_seconds: float=0.0, mode: str='and', debug: bool=False, optimize: bool=True
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.detect_texts_from_video(case['args']) == case['expected']

def test_ocr_on_roi():
    cases = [
                {
                'args': {
                    #frame, t, debug
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert detect_text.ocr_on_roi(case['args']) == case['expected']

