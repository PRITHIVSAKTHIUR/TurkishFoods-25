import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

model_name = "prithivMLmods/TurkishFoods-25"  # Replace with your Hugging Face repo
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

id2label = {
    "0": "asure", "1": "baklava", "2": "biber_dolmasi", "3": "borek", "4": "cig_kofte",
    "5": "enginar", "6": "et_sote", "7": "gozleme", "8": "hamsi", "9": "hunkar_begendi",
    "10": "icli_kofte", "11": "ispanak", "12": "izmir_kofte", "13": "karniyarik", "14": "kebap",
    "15": "kisir", "16": "kuru_fasulye", "17": "lahmacun", "18": "lokum", "19": "manti",
    "20": "mucver", "21": "pirinc_pilavi", "22": "simit", "23": "taze_fasulye", "24": "yaprak_sarma"
}

def predict_food(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    return {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}

iface = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Top Turkish Foods"),
    title="TurkishFoods-25 Classifier",
    description="Upload a food image to identify one of 25 Turkish dishes."
)

if __name__ == "__main__":
    iface.launch()
