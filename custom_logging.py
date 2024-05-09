import logging
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="huggingface_hub")

ollm_logging = logging.getLogger("OpenLLM")
ollm_logging.setLevel(logging.INFO)
ollm_logging.propagate = False

ollm_logging_sh = logging.StreamHandler()
ollm_logging_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
ollm_logging_sh.setLevel(logging.INFO)
ollm_logging.addHandler(ollm_logging_sh)
