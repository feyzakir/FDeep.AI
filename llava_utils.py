from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

def load_llava_model():
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return processor, model

def analyze_image(processor, model, image: Image.Image, question: str):
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200)
    output = processor.decode(generated_ids[0], skip_special_tokens=True)
    return output
