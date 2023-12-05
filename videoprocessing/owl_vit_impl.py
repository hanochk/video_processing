import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection


from nebula3_videoprocessing.videoprocessing.vg_interface import VgInterface
from typing import NewType
ScoredBbox = NewType('ScoredBbox', tuple[list, float])

class OwlVitImplementation(VgInterface):

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device=self.device)


    def ground_objects(self, image, text): #TODO: Check if this works as expected, use batch one for now

        texts = [[text]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(device=self.device)
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]]).to(device=self.device)
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        score_threshold = 0.1
        scored_bbox = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                scored_bbox.append((box, round(score.item(), 3)))


        return list(ScoredBbox(scored_bbox))

    def ground_objects_batch(self, image, texts):
        with torch.no_grad():
            inputs = self.processor(text=texts, images=image, return_tensors="pt").to(device=self.device)
            outputs = self.model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1]]).to(device=self.device)
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            score_threshold = 0.1
            scored_bbox = [[] for i in range(len(texts))]

            # idx = 0
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= score_threshold:
                    scored_bbox[int(label)].append((box, round(score.item(), 3)))
                    cur_text = texts[int(label)]
                    # print(f"Detected {cur_text} with confidence {round(score.item(), 3)} at location {box}")

            return list(list(ScoredBbox(scored_bbox)))


def main():
    ### OWL VIT USAGE EXAMPLE ###
    owl_vit = OwlVitImplementation()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "a photo of a remote"
    ScoredBbox_ = owl_vit.ground_objects(image, text)

    print("------------------------------------------------------")

    texts = [["a photo of a remote"], ["a photo of a cat"]]
    ScoredBbox = owl_vit.ground_objects_batch(image, texts)
    a=0

if __name__ == "__main__":
    main()
    pass