
from fcp_ocr import place_markers

def test_place():
    cases = [
                {
                'args': {
                    #root, texts: list[dict], fps: str, keyword: str, in_event: bool=False
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert place_markers.place(case['args']) == case['expected']

