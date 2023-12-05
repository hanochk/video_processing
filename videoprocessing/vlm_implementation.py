from nebula3_videoprocessing.videoprocessing.vlm_interface import VlmInterface
from nebula3_videoprocessing.videoprocessing.utils.config import config
import typing
from PIL import Image
import requests
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from videoprocessing.models.blip_itm import blip_itm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# from nebula3_experts_vg.vg.visual_grounding_inference import OfaMultiModalVisualGrounding
# from nebula3_videoprocessing.videoprocessing.owl_vit_impl import OwlVitImplementation


class VlmBaseImplementation(VlmInterface):

    def compute_similarity_url(self, url: str, text: list[str]):
        image = self.load_image_url(url)
        return self.compute_similarity(image, text)
    
class VlmChunker(VlmBaseImplementation):
    def __init__(self, vlm: VlmInterface, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.vlm = vlm

    def load_image_url(self,url):
        return self.vlm.load_image_url(url)
    
    def compute_similarity(self, image: Image, text: list[str]) -> list[float]:
        results = []
        chunked_texts=[text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        for chunk in chunked_texts:
            results.extend(self.vlm.compute_similarity(image,chunk))
        return results    

class VisualGroundingToVlmAdapter(VlmBaseImplementation):

    def __init__(self): # vg : VgInterface
        self.vg = OwlVitImplementation()
    
    def compute_similarity(self, image, text):
        scored_bboxes = self.vg.ground_objects_batch(image, text)
        if not scored_bboxes:
            return []

        scores = []
        for idx in range(len(scored_bboxes) - 1):
            if not scored_bboxes[idx]:
                scores.append(0.0)
                continue
            else:
                max_conf = sorted(scored_bboxes[idx],key=lambda x: x[1], reverse=True)[0][1]#max([scored_bboxes[idx]], key=lambda item:item[1])[1]
                scores.append(max_conf)

        return scores






class ClipVlmImplementation(VlmBaseImplementation):

    def __init__(self, init_with_cpu=False):

        if init_with_cpu:
            print("Initializing model on CPU")
            self.device = torch.device('cpu')
        else:
            print("Initializing model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CLIPModel.from_pretrained(config["clip_checkpoints"]).to(device=self.device)
        self.processor = CLIPProcessor.from_pretrained(config["clip_checkpoints"])


    def load_image_url(self, url: str):
        return Image.open(requests.get(url, stream=True).raw)  

    def compute_similarity(self, image : Image, text : list[str]):

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(device=self.device)

        outputs = self.model(**inputs)
        embeds_dotproduct = (outputs.image_embeds.expand_as(outputs.text_embeds) * outputs.text_embeds).sum(dim=1)
        return embeds_dotproduct.cpu().detach().numpy()

class BlipItmVlmImplementation(VlmBaseImplementation):
    def __init__(self, init_with_cpu = False):

        if init_with_cpu:
            print("Initializing model on CPU")
            self.device = torch.device('cpu')
        else:
            print("Initializing model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = blip_itm(pretrained=config['blip_model_url_large'], image_size=config['blip_image_size'], vit=config['blip_vit_large'])
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
        return image

    def load_image(self, image: Image): 
        transform = transforms.Compose([
            transforms.Resize((config['blip_image_size'], config['blip_image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(image).unsqueeze(0).to(self.device)   
        return image

    def compute_similarity(self, image: Image, text: list[str]):
        
        image = self.load_image(image)

        itm_output = self.model(image, text, match_head='itm')
        # Change from softmax to dotproduct
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        itm_scores = itm_score.cpu().detach().numpy()

        return itm_scores


class BlipItcVlmImplementation(VlmBaseImplementation):
    def __init__(self, init_with_cpu = False):
        if init_with_cpu:
            print("Warning: Initializing BLIP_ITC model on CPU")
            self.device = 'cpu'
        else:
            print("Warning: Initializing BLIP_ITC model on GPU")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = blip_itm(pretrained=config['blip_model_url_large'], image_size=config['blip_image_size'], vit=config['blip_vit_large'])
        model.eval()
        self.model = model.to(device=self.device)
    
    def load_image_url(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB') 
        return image

    def load_image(self, image):   
        
        transform = transforms.Compose([
            transforms.Resize((config['blip_image_size'], config['blip_image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(image).unsqueeze(0).to(self.device)   
        return image

    def compute_similarity(self, image : Image, text : list[str]):
        image = self.load_image(image)
        itc_output = self.model(image, text, match_head='itc')
        # Check if its dotproduct
        itc_scores = itc_output.cpu().detach().numpy()[0]
        return itc_scores
    
    # def bbox_xywh_to_xyxy(self, xywh):
    #     w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
    #     return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    
    def crop_image(self, image : Image, bbox: list[float]):
        # xmin, ymin, xmax, ymax = self.bbox_xywh_to_xyxy((bbox[0],bbox[1],bbox[2],bbox[3]))
        cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # cropped_image.save("/notebooks/test123.jpg")
        return cropped_image
    
    def compute_similarity_on_bboxes(self, image : Image, text : list[str], bbox : list[float]):

        cropped_image = self.crop_image(image, bbox)
        return self.compute_similarity(cropped_image, text)
    
class VisualGroundingVlmImplementation(VlmInterface):
        def __init__(self):
            self.vg_engine = OfaMultiModalVisualGrounding()
        
        def load_image(self):
            pass

        def compute_similarity(self, image : Image, text : list[str]):
            time_measure = False
            if time_measure:
                import time
                since = time.time()

            bb, _, lprob = self.vg_engine.find_visual_grounding(image, text)

            if time_measure:
                time_elapsed = time.time() - since
                print('OFA VG time {:.3f}s'.format(time_elapsed))

            lprob = lprob.sum()
            debug = False
            if debug:
                plot_vg_over_image(bb, image, caption=text, lprob=lprob)

            return bb, lprob.cpu().numpy()


def main():
    ### CLIP USAGE EXAMPLE ###
    clip_vlm = ClipVlmImplementation()

    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    text =['a woman sitting on the beach with a dog', 'a man standing on the beach with a cat']
    similarity = clip_vlm.compute_similarity_url(url, text)
    print(f"CLIP outputs: {similarity}")

    ##################################

    ### BLIP ITM USAGE EXAMPLE ###

    blip_vlm = BlipItmVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    similarity = blip_vlm.compute_similarity_url(url, text)
    itm_score = similarity
    print(f"BLIP_ITM outputs:")
    print('The image and text is matched with a probability of %.4f'%itm_score)

    ## BLIP ITC USAGE EXAMPLE ###
    blip_vlm = BlipItcVlmImplementation()
    text = ['a woman sitting on the beach with a dog']
    similarity = blip_vlm.compute_similarity_url(url, text)
    itc_score = similarity
    print(f"BLIP_ITC outputs:")
    print('The image and text is matched with a probability of %.4f'%itc_score)




if __name__ == "__main__":
    # main()
    pass