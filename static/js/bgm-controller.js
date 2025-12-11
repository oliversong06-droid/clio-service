(function (global) {
    const STORAGE_KEYS = {
        time: 'bgmTime',
        intent: 'bgmIntent'
    };
    const DEFAULTS = {
        volume: 0.5,
        autoStart: true,
        respectIntent: true,
        persistInterval: 1600
    };

    const safeGet = (key) => {
        try {
            return sessionStorage.getItem(key);
        } catch (err) {
            return null;
        }
    };

    const safeSet = (key, value) => {
        try {
            sessionStorage.setItem(key, value);
        } catch (err) {}
    };

    const parseSeconds = (value) => {
        if (typeof value !== 'string' || value.trim() === '') {
            return null;
        }
        const time = parseFloat(value);
        return Number.isFinite(time) && time >= 0 ? time : null;
    };

    function initController(target, options) {
        const audio = typeof target === 'string' ? document.getElementById(target) : target;
        if (!audio) {
            return null;
        }

        const settings = Object.assign({}, DEFAULTS, options || {});
        audio.volume = settings.volume;
        const preloadValue = (audio.getAttribute('preload') || '').toLowerCase();
        if (!preloadValue || preloadValue === 'none') {
            audio.setAttribute('preload', 'auto');
        }
        if (audio.preload === 'auto' && audio.readyState === 0) {
            try {
                audio.load();
            } catch (err) {}
        }

        const markIntent = (isPlaying) => safeSet(STORAGE_KEYS.intent, isPlaying ? '1' : '0');
        const hasIntent = () => safeGet(STORAGE_KEYS.intent) === '1';

        const applySavedTime = () => {
            const saved = parseSeconds(safeGet(STORAGE_KEYS.time));
            if (saved == null) {
                return;
            }
            const updateTime = () => {
                try {
                    const duration = audio.duration;
                    const maxTime = Number.isFinite(duration) && duration > 0 ? Math.min(saved, Math.max(0, duration - 0.25)) : saved;
                    audio.currentTime = maxTime;
                } catch (err) {}
            };
            if (audio.readyState >= 1) {
                updateTime();
            } else {
                audio.addEventListener('loadedmetadata', updateTime, { once: true });
            }
        };

        applySavedTime();

        const persistTime = () => {
            const time = audio.currentTime;
            if (Number.isFinite(time)) {
                safeSet(STORAGE_KEYS.time, String(time));
            }
        };

        const startPersistence = () => {
            if (settings.persistInterval <= 0) {
                return;
            }
            setInterval(persistTime, settings.persistInterval);
            ['visibilitychange', 'pagehide', 'beforeunload'].forEach((eventName) => {
                window.addEventListener(eventName, () => {
                    if (eventName !== 'visibilitychange' || document.hidden) {
                        persistTime();
                    }
                });
            });
        };

        startPersistence();

        let gesturePending = false;
        const removeGestureListeners = () => {
            document.removeEventListener('pointerdown', onGesture);
            document.removeEventListener('keydown', onGesture);
        };
        const onGesture = () => {
            gesturePending = false;
            removeGestureListeners();
            attemptPlay(false);
        };
        const requestGesture = () => {
            if (gesturePending) {
                return;
            }
            gesturePending = true;
            document.addEventListener('pointerdown', onGesture, { once: true });
            document.addEventListener('keydown', onGesture, { once: true });
        };

        const attemptPlay = (respectIntent = settings.respectIntent) => {
            if (respectIntent && !hasIntent()) {
                return Promise.resolve(false);
            }
            const playback = audio.play();
            const handleSuccess = () => {
                gesturePending = false;
                removeGestureListeners();
                markIntent(true);
                return true;
            };
            const handleFailure = (err) => {
                requestGesture();
                if (settings.debug && err) {
                    console.warn('BGM autoplay blocked', err);
                }
                return false;
            };
            if (playback && typeof playback.then === 'function') {
                return playback.then(handleSuccess).catch(handleFailure);
            }
            return Promise.resolve(handleSuccess());
        };

        const autoStart = () => {
            const startPlayback = () => attemptPlay(settings.respectIntent);
            startPlayback();
            if (audio.readyState >= 2) {
                return;
            }
            const events = ['loadeddata', 'canplay', 'canplaythrough'];
            const onReady = () => {
                events.forEach((eventName) => audio.removeEventListener(eventName, onReady));
                startPlayback();
            };
            events.forEach((eventName) => audio.addEventListener(eventName, onReady));
            const onVisible = () => {
                if (!document.hidden) {
                    document.removeEventListener('visibilitychange', onVisible);
                    startPlayback();
                }
            };
            document.addEventListener('visibilitychange', onVisible);
        };

        if (settings.autoStart) {
            autoStart();
        }

        audio.addEventListener('play', () => markIntent(true));
        audio.addEventListener('pause', () => markIntent(false));
        window.addEventListener('pageshow', () => attemptPlay(false));

        return {
            audio,
            play: (respectIntent) => attemptPlay(typeof respectIntent === 'boolean' ? respectIntent : false),
            ensure(respectIntent) {
                return attemptPlay(typeof respectIntent === 'boolean' ? respectIntent : settings.respectIntent);
            },
            requestGesture,
            applySavedTime,
            persistTime,
            hasIntent,
            markIntent
        };
    }

    global.BgmController = {
        init: initController,
        keys: STORAGE_KEYS
    };
})(window);
