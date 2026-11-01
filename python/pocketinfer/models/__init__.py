"""Model client helpers.

The three bhashini clients (asr, nmt, tts) all talk to one Flask service on
port 11400. Their readiness check belongs here rather than copied into each of
them, because it has to be shared: whichever verifies first pays the wait, and
a dead service must not cost three full timeouts in series before the app
starts.
"""
import logging
import time
from subprocess import check_output

import requests

HEALTH_URL = "http://localhost:11400/health"

# How long to wait before concluding the service needs a kick. Measured on this
# board: bhashini_models takes 40-60s from systemd start to its first healthy
# /health, because it loads the ASR, both NMT directions and the TTS voices.
#
# This was 15s, and on a cold boot that restarted a service which was 36 seconds
# into loading and would have been ready shortly after -- throwing the work away
# and starting the clock again. Wait generously; restart only as a last resort.
SETTLE_WAIT = 90.0

# After a restart it loads from scratch, so allow the same again.
RESTART_WAIT = 90.0

# How long a health verdict stays good for. Only has to span one pass of
# verify_dependencies across the three clients.
_MEMO_SECONDS = 15.0

_last = (0.0, None)   # (when, (ok, message))


def wait_healthy(seconds, url=HEALTH_URL):
    """Poll until the service answers, or the time runs out. Never raises."""
    deadline = time.time() + seconds
    while True:
        try:
            if requests.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(0.25)


def ensure_healthy():
    """Make the bhashini service ready, or report why not. Memoised.

    Returns (ok, message). Restarting is the last resort, not the first move:
    the systemd unit orders the app After=bhashini_models.service, so at boot
    the service is always mid-load when we first look.
    """
    global _last
    when, verdict = _last
    if verdict is not None and time.time() - when < _MEMO_SECONDS:
        return verdict

    if wait_healthy(SETTLE_WAIT):
        _last = (time.time(), (True, "bhashini service is available."))
        return _last[1]

    logging.warning("bhashini did not answer in %.0fs; restarting it", SETTLE_WAIT)
    try:
        check_output("systemctl restart bhashini_models.service", shell=True)
    except Exception as e:
        _last = (time.time(), (False, "bhashini is down and could not be restarted: %s" % e))
        return _last[1]

    if wait_healthy(RESTART_WAIT):
        _last = (time.time(), (True, "bhashini service is available."))
    else:
        # Was `response.status_code`, on a path where `response` is never bound
        # if every attempt raised ConnectionError -- so the one branch that
        # exists to report a dead service raised UnboundLocalError instead.
        _last = (time.time(), (False, "bhashini did not become healthy within %.0fs." % RESTART_WAIT))
    return _last[1]
