
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llama_cpp import Llama


def load_MPropositioneur_V2():
    model_id = "Zual/MPropositioneur-V2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("mps" if torch.backends.mps.is_available() else "gpu" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    return model, tokenizer

def load_Qwen3_8B_via_llama_cpp():
    model = Llama(
    model_path="models/Qwen3-8B-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False)

    return model
