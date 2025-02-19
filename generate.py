import torch
from transformers import pipeline

model_id =  "tpo-alignment/Llama-3-8B-TPO-40k"

generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
outputs = generator([{"role": "user", "content": "calculate 2+2."}], do_sample=False, max_new_tokens=200)
print(outputs[0]['generated_text'])