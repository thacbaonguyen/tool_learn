COMPOSE ?= docker compose
VIDEO ?= data_sample/016 IAM Security Tools.mp4
VI_SRT ?= data_sample/vi_sample.srt
VOICE_START ?= 0
VOICE_DURATION ?= 10

.PHONY: up down logs voice build build-web build-tts download-tts-model test test-web web ocr evaluate tts render report phase0

build-web:
	$(COMPOSE) build web

up:
	$(COMPOSE) up -d web worker

web: up

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f web worker

voice:
	$(COMPOSE) run --rm --no-deps worker python -m app.voice_setup --start "$(VOICE_START)" --duration "$(VOICE_DURATION)"

test-web:
	$(COMPOSE) run --rm --no-deps --entrypoint pytest web tests/test_app.py -q

build:
	$(COMPOSE) --profile phase0 build phase0

build-tts:
	$(COMPOSE) --profile phase0 build tts

download-tts-model: build-web
	$(COMPOSE) --profile setup run --rm model-setup

test:
	$(COMPOSE) --profile phase0 run --rm --entrypoint pytest phase0 -q

ocr:
	$(COMPOSE) --profile phase0 run --rm phase0 poc.extract_subtitles --input "$(VIDEO)" --output artifacts/english_ocr.srt

evaluate:
	$(COMPOSE) --profile phase0 run --rm phase0 poc.evaluate_ocr --predicted artifacts/english_ocr.srt --reference artifacts/ground_truth.srt

tts:
	$(COMPOSE) --profile phase0 run --rm tts poc.tts --video "$(VIDEO)" --srt "$(VI_SRT)" --provider vtts --output artifacts/tts_timeline.wav

render:
	$(COMPOSE) --profile phase0 run --rm phase0 poc.render --video "$(VIDEO)" --srt "$(VI_SRT)" --audio artifacts/tts_timeline.wav --output artifacts/phase0_preview.mp4

report:
	$(COMPOSE) --profile phase0 run --rm phase0 poc.report --artifacts artifacts --output docs/phase0-report.md

phase0: build build-tts test ocr evaluate tts render report
