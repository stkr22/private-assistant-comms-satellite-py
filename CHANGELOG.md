# Changelog

## [1.1.0](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v1.0.0...v1.1.0) (2025-08-07)


### Features

* :sound: add programmatic sweep sound generation [AI] ([d4ac0e6](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/d4ac0e6dae497fbbb37aaf8b3a1ab51db139ed45))
* :sound: add programmatic sweep sound generation [AI] ([ec67ba4](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/ec67ba44a2a2e249248b0b850906ee22ff179147))

## [1.0.0](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v0.6.0...v1.0.0) (2025-08-06)


### ⚠ BREAKING CHANGES

* PyAudio optional dependency removed - sounddevice is now core audio library

### Features

* :loud_sound: migrate from PyAudio to sounddevice [AI] ([2986c7b](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/2986c7b9788e4bdb85084e6e8a2046d668ab07bb))
* migrate from PyAudio to sounddevice for improved audio processing ([a785352](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/a78535267289a1cc364b752945df43b6a6445d32))


### Documentation

* add concise getting started guide for edge device setup [AI] ([7ff1851](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/7ff1851205ba8d9255753183b6d4948672ae02a6))
* add concise getting started guide for edge device setup [AI] ([d197b0f](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/d197b0fb75a3576507be2c631d11ae07f7136ab8))

## [0.6.0](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v0.5.0...v0.6.0) (2025-07-31)


### Features

* add automatic OpenWakeWord model download on first startup [AI] ([1f6803f](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/1f6803f34e6f2d27a47de56d4369e1c79a15c2dd))
* add WebSocket and SSL support for MQTT connections [AI] ([21fe75b](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/21fe75b005f94e3f7e970c3a6ad1726c3f97153d))
* configure PyAudio as optional dependency for CI compatibility [AI] ([a6924bd](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/a6924bdfd4c147902e44e8643a5470daff17478e))


### Bug Fixes

* update CI to exclude PyAudio extra [AI] ([75d7bcc](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/75d7bccff5f78fdbe2b99ad3261f4e999bd8218b))

## [0.4.0](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v0.3.0...v0.4.0) (2025-07-29)


### Features

* optimize memory with pre-allocated audio buffers [AI] ([7b82b68](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/7b82b682fafe5dcbc158a0ca19f9563b51a3db7e))
* optimize memory with pre-allocated audio buffers [AI] ([d3416c0](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/d3416c006e8c8a122d385191a8ea0b249caa4d17)), closes [#9](https://github.com/stkr22/private-assistant-comms-satellite-py/issues/9)
* vectorized VAD processing [AI] ([feded86](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/feded86df892a1400f6de36bdf6e78231a661484))
* vectorized VAD processing [AI] ([9ad3ccf](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/9ad3ccf0ad6a38609a7a5016fb9c4770e48f02fb))

## [0.3.0](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v0.2.0...v0.3.0) (2025-07-29)


### Features

* implement 3-thread architecture [AI] ([0cab4f6](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/0cab4f648348621c9d96ae87d0aa61d967841d13))

## [0.2.0](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v0.1.1...v0.2.0) (2025-07-29)


### Features

* refactor to async/await architecture [AI] ([a53a95a](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/a53a95ad3800d05d44da1a8ae9c03d2776457f8d))
* refactor to async/await architecture [AI] ([7c704d9](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/7c704d9b13e48ca209ac5b4b4e4c65c25286408a)), closes [#7](https://github.com/stkr22/private-assistant-comms-satellite-py/issues/7)


### Bug Fixes

* implementing correct parsing of mqtt payloads ([ffc09db](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/ffc09db9f2ef647e6279d663d54e69dbf10706c8))
* remove blocking API calls from audio thread [AI] ([59d8e38](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/59d8e38795058cd9cce66d95a1b02af3380ecfbb)), closes [#6](https://github.com/stkr22/private-assistant-comms-satellite-py/issues/6)
* resolve ruff linting issues [AI] ([2806eb2](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/2806eb2f51c1fdcf02492239befd7db93a317b8f))


### Documentation

* comprehensive documentation update with performance focus [AI] ([fd55268](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/fd5526818635656c28ec31c6a2dd3787ca8b9a43))

## [0.1.1](https://github.com/stkr22/private-assistant-comms-satellite-py/compare/v0.1.0...v0.1.1) (2025-07-29)


### Bug Fixes

* adding inference framework as parameter ([64f55a0](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/64f55a010dc49944db99ab4d8c49fb582e701452))
* adding inference framework as parameter ([f473269](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/f4732692d67de9c5a7811426f9ec25976b5a19fc))

## 0.1.0 (2025-07-28)


### Features

* add CLI interface and remove backward compatibility [AI] ([72f7a71](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/72f7a7113e55cf89ca92a3bc35db0e528e70bd51))


### Bug Fixes

* make pyaudio optional dependency for CI/CD compatibility [AI] ([04cc12c](https://github.com/stkr22/private-assistant-comms-satellite-py/commit/04cc12ce86aac005b2b78b9599b5124446c54946))
