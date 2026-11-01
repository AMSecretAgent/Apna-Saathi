import ollama
import logging
import time
from subprocess import check_output
import requests


class Ollama:
    def __init__(self, model_name: str):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        #TODO execute curl http://localhost:11434/api/generate -d '{"model": model_name, "keep_alive": -1}'

    def chat(self, messages: list) -> ollama.ChatResponse:
        return ollama.chat(model=self.model_name, messages=messages)

    def generate(self, prompt, images=None, system=None, options=None, keep_alive=-1, think=None):
        # keep_alive defaults to -1 (never unload). Ollama's own default is a
        # 5 minute idle timeout, and it is reset by *every* request that does not
        # say otherwise -- so a single generate() without this silently undoes the
        # warm hold that verify() sets up, and the next question after a quiet
        # spell pays a 10-30s model reload. On camera that reads as a hang.
        kwargs = {"model": self.model_name, "prompt": prompt, "keep_alive": keep_alive}
        if images is not None:
            kwargs["images"] = images
        if system is not None:
            kwargs["system"] = system
        if options is not None:
            kwargs["options"] = options
        # Qwen3 reasons in a <think> block before answering unless told not
        # to; on a 1.7B model that is seconds of tokens nobody hears. None
        # leaves the model's default, so older models are unaffected.
        if think is not None:
            kwargs["think"] = think
        return ollama.generate(**kwargs)

    def restart(self):
        print(check_output('systemctl restart ollama', shell=True))
        requests.post('http://localhost:11434/api/generate', json={'model': self.model_name, 'keep_alive': -1})
    
    @classmethod
    def verify(cls, args):
        # Wait for the daemon before deciding anything.
        #
        # This used to be a bare `except Exception: return False`, which cannot
        # tell "ollama.service has not finished starting" from "the model is
        # missing". The caller's response to False is update() -> ollama.pull(),
        # and this device is demonstrated with the wifi switched off, so that
        # pull fails and verify_dependencies raises -- killing the process
        # before the button ever works. On a cold boot that is the whole demo.
        #
        # So: poll first, and only report False for a model that is genuinely
        # absent, which is the one case where pulling is the right answer.
        deadline = time.time() + 45.0
        listed = None
        while True:
            try:
                listed = ollama.list()
                break
            except Exception:
                if time.time() >= deadline:
                    break
                time.sleep(0.5)

        if listed is None:
            try:
                check_output('systemctl restart ollama', shell=True)
            except Exception:
                logging.exception("could not restart ollama")
            return False, "Ollama daemon did not answer within 45s."

        for model in listed.models:
            if model.model == args["model_name"]:
                # Preload and pin. This can take 10-30s cold, so it gets a real
                # timeout -- but a preload that does not finish is not a reason
                # to call the model missing and go looking for a network. The
                # model is on disk; the first question just pays the load.
                # The pin must carry the same options the application will
                # generate with. Ollama keys its runner on num_ctx and num_gpu:
                # a pin with defaults and a first generate with num_ctx=1024
                # loads the model twice at boot, and each load re-fits to
                # whatever memory is free at that moment.
                try:
                    requests.post('http://localhost:11434/api/generate',
                                  json={'model': args['model_name'], 'keep_alive': -1,
                                        'options': args.get('options', {})},
                                  timeout=120)
                except Exception:
                    logging.warning("model %s is present but did not preload; "
                                    "the first question will pay the load",
                                    args['model_name'])
                return True, "Ollama service is available."

        return False, f"Model '{args['model_name']}' not found."

    @classmethod
    def update(cls, args):
        # Ollama models are managed by the Ollama service itself.
        # This method can be used to pull the latest model if needed.
        # TODO - ensure ollama service is running
        try:
            logging.info(f"Pulling Ollama model '{args['model_name']}'")
            # Bounded. The module-level client has no timeout, so wifi that
            # is associated but has no route makes this block forever -- and
            # a hang cannot be caught by the guard in ApnaSaathi.
            ollama.Client(timeout=30).pull(model=args["model_name"])
        except Exception as e:
            raise RuntimeError(f"Failed to update Ollama model '{args['model_name']}': {str(e)}")
