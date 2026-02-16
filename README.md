# fcp-ocr
Optical Character Recognition (OCR) in FCPXML

This module detects texts in desired regions of interests (ROI) and place Markers in FCPXML.

Sample command:
`fcp-ocr --targetp Save:48,45.5,52,48 --targetp Photo:45,39,55,42 --skip_seconds=1.5 --ocr_mode=or --affix=ocr_marked_ --debug=0 --optimize=1 --sync=1 <filepath>`

This is an initial commit of the developing project.
