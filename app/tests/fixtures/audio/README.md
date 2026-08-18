# Audio integration fixture

`vad_speech_then_silence.s16le.pcm` contains 81 frames of mono, 16 kHz,
signed 16-bit little-endian PCM (512 samples / 1,024 bytes per frame). It is
the first 82,944 bytes of the repository's existing
`test_speaker/man_enroll.s16le.pcm` test recording.

The fixture includes speech followed by enough silence for the backend VAD
state machine to emit exactly one speech-start and one speech-end boundary.
Its SHA-256 is
`365b7ae07e2f8fecd3c4e3f0bbe49ace6d1d096eeaea0fcdede626e2970da5d0`.
