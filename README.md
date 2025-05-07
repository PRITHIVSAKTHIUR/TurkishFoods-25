![4.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_hdtIoG6zxJk7WSwmSFHh.png)

# TurkishFoods-25

> **TurkishFoods-25** is a computer vision model fine-tuned from `google/siglip2-base-patch16-224` for multi-class food image classification. It is trained to identify 25 traditional Turkish dishes using the `SiglipForImageClassification` architecture.

```py
Classification Report:
                precision    recall  f1-score   support

         asure     0.9718    0.9503    0.9609       181
       baklava     0.9589    0.9292    0.9438       452
 biber_dolmasi     0.9505    0.9555    0.9530       382
         borek     0.8770    0.8842    0.8806       613
     cig_kofte     0.9051    0.9358    0.9202       265
       enginar     0.9116    0.8753    0.8931       377
       et_sote     0.7870    0.7688    0.7778       346
       gozleme     0.9220    0.9420    0.9319       414
         hamsi     0.9724    0.9763    0.9744       253
hunkar_begendi     0.9583    0.9274    0.9426       248
    icli_kofte     0.9261    0.9353    0.9307       402
       ispanak     0.9567    0.9343    0.9454       213
   izmir_kofte     0.8763    0.9239    0.8995       368
    karniyarik     0.9538    0.8934    0.9226       347
         kebap     0.9154    0.8584    0.8860       706
         kisir     0.8919    0.9356    0.9132       388
  kuru_fasulye     0.8799    0.9820    0.9281       388
      lahmacun     0.9699    0.8703    0.9174       185
         lokum     0.9220    0.9369    0.9294       555
         manti     0.9569    0.9482    0.9525       328
        mucver     0.8743    0.9201    0.8966       363
 pirinc_pilavi     0.9110    0.9482    0.9292       367
         simit     0.9629    0.9284    0.9453       391
  taze_fasulye     0.8992    0.9253    0.9121       241
  yaprak_sarma     0.9742    0.9544    0.9642       395

      accuracy                         0.9186      9168
     macro avg     0.9234    0.9216    0.9220      9168
  weighted avg     0.9194    0.9186    0.9186      9168
```

---

## Label Space: 25 Classes

The model classifies food images into the following Turkish dishes:

```json
"id2label": {
  "0": "asure",
  "1": "baklava",
  "2": "biber_dolmasi",
  "3": "borek",
  "4": "cig_kofte",
  "5": "enginar",
  "6": "et_sote",
  "7": "gozleme",
  "8": "hamsi",
  "9": "hunkar_begendi",
  "10": "icli_kofte",
  "11": "ispanak",
  "12": "izmir_kofte",
  "13": "karniyarik",
  "14": "kebap",
  "15": "kisir",
  "16": "kuru_fasulye",
  "17": "lahmacun",
  "18": "lokum",
  "19": "manti",
  "20": "mucver",
  "21": "pirinc_pilavi",
  "22": "simit",
  "23": "taze_fasulye",
  "24": "yaprak_sarma"
}
```

---

## Install Requirements

```bash
pip install -q transformers torch pillow gradio
```

---

## Inference Script

```python
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
```

---

## Applications

* Turkish cuisine image datasets
* Food delivery or smart restaurant apps
* Culinary learning platforms
* Nutrition tracking via image-based recognition 
