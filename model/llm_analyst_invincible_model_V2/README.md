---
base_model: meta-llama/Llama-3.1-8B
library_name: peft
---

# LLM Analyst — Fine-tuned Solar Flare Explainer

This is a LoRA adapter for Llama 3 8B, fine-tuned to generate 
natural language explanations of solar flare predictions.

## Model Description

- **Base model:** meta-llama/Llama-3.1-8B
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via PEFT
- **Task:** Explainable AI for solar flare forecasting
- **Skills:** Explanation, Recommendation, Counterfactual
- **Training data:** 50,533 synthetically generated instruction-response pairs
- **Evaluation:** 8.04/10 Faithfulness score (Gemini 2.5 Pro judge, n=200)

## Training Details

- **Stage 1:** Explanation skill only (LR = 2.98e-4, 1 epoch)
- **Stage 2:** All three skills (LR = 1.49e-4, 1 epoch)
- **LoRA rank:** 32, Alpha: 128, Dropout: 0.066
- **Hardware:** NVIDIA A100 GPU

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./").merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
```

## Citation

```bibtex
@inproceedings{chaudhary2026llmflare,
  title={Explainable Solar Flare Prediction Using Deep Learning 
         and Large Language Models},
  author={Chaudhary, Yash and Abdullah, Yasser and Wang, Jason T.L.},
  booktitle={Proceedings of ICTAI 2026},
  year={2026}
}
```

## License
MIT License