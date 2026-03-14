
from fcp_ocr import cli

def test_get_resolution():
    cases = [
                {
                'args': {
                    #filepath
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert cli.get_resolution(case['args']) == case['expected']

def test_parse_targetp():
    cases = [
                {
                'args': {
                    #targetp, width, height
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert cli.parse_targetp(case['args']) == case['expected']

def test_main():
    cases = [
                {
                'args': {
                    #
                    },
                #
                'expected': ,
                },
            ]

    for case in cases:
        assert cli.main(case['args']) == case['expected']

