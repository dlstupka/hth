Workflow architecture proposal.

Reusable workflow:
- _core-hth.yml

Entry workflows:
- preprocess.yml
- preprocess-test.yml
- calibrate-geometry.yml

Preferred stage names:
stage_preprocess
stage_detect
stage_detect_<algorithm>
stage_validate
stage_ocr
stage_transcribe
stage_translate_<lang>[_<archaic>]
stage_extract
stage_reason
stage_publish
