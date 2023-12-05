from nebula3_videoprocessing.videoprocessing.ontology_interface import OntologyInterface
from nebula3_videoprocessing.videoprocessing.ontology_factory import OntologyFactory
from nebula3_videoprocessing.videoprocessing.vlm_factory import VlmFactory
from nebula3_videoprocessing.videoprocessing.utils import consts
import typing
from PIL import Image
import requests
import torch


# DUMMY_IMAGE = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
EMBBDING_BATCH_LIMIT_TEXT = 512

def get_prefix_prompt_functions():
            attribute_prompt = lambda x: f'A photo of {x}'
            scene_prompt = lambda x: f'A photo of {x}'
            verb_prompt = lambda x: f'A photo of {x}'
            object_prompt = lambda x: f'A photo of {x}'
            vg_attribute_prompt = lambda x: f'A photo of something or somebody {x}'
            persons_prompt = lambda x: f'A photo of {x}'
            vg_verb_prompt = lambda x: f'A photo of something capable of {x}'
            indoor_prompt = lambda x: f'A photo of {x}'
            return {
                'objects':object_prompt,
                'vg_objects':object_prompt,
                'attributes':attribute_prompt,
                'vg_attributes': vg_attribute_prompt,
                'scenes':scene_prompt,
                'persons': persons_prompt,
                'verbs':verb_prompt,
                'vg_verbs': vg_verb_prompt
                #'indoors': indoor_prompt
            }

class SingleOntologyImplementation(OntologyInterface):
    def __init__(self, ontology_name : str, vlm_name : str):

        vlm_factory = VlmFactory()
        ontology_factory = OntologyFactory()

        self.vlm = vlm_factory.get_vlm(vlm_name)
        self.ontology = ontology_factory.get_ontology(ontology_name)
        self.ontology_name = ontology_name
        self.prompt_functions = get_prefix_prompt_functions()
        
        # for key in consts.OMIT_KEYWORDS:
        #     if key in self.ontology: self.ontology.remove(key)

        self.texts = [self.prompt_functions[self.ontology_name](t) for t in self.ontology]
        print(f"Length of ontology: {len(self.texts)}")
    

    def compute_scores(self, image) -> list[(str, float)]:
        
        outputs = []

        texts = self.texts

        # If VLM crashes, you can extend 10 to bigger number.
        div_texts = len(texts) // 20
        len_texts = len(texts) 
        for i in range(0, len_texts, div_texts):
            if (i + div_texts) > len_texts:
                scores = self.vlm.compute_similarity(image, texts[i:i + (len_texts - i)])
                for j, score in enumerate(scores):
                    outputs.append((self.ontology[i + j], score))
            else:
                scores = self.vlm.compute_similarity(image, texts[i:i + div_texts])
                for j, score in enumerate(scores):
                    outputs.append((self.ontology[i + j], score))
        
        # outputs.sort(key=lambda a: a[1], reverse=True)
        # print(f"Top 5: {outputs[:5]}")
        return outputs
    
    def compute_scores_with_bboxes(self, image, bbox) -> list[(str, float)]:
        
        outputs = []

        texts = self.texts

        # If VLM crashes, you can extend 10 to bigger number.
        div_texts = len(texts) // 20
        len_texts = len(texts) 
        for i in range(0, len_texts, div_texts):
            if (i + div_texts) > len_texts:
                scores = self.vlm.compute_similarity_on_bboxes(image, texts[i:i + (len_texts - i)], bbox)
                for j, score in enumerate(scores):
                    outputs.append((self.ontology[i + j], score))
            else:
                scores = self.vlm.compute_similarity_on_bboxes(image, texts[i:i + div_texts], bbox)
                for j, score in enumerate(scores):
                    outputs.append((self.ontology[i + j], score))
        
        # outputs.sort(key=lambda a: a[1], reverse=True)
        # print(f"Top 5: {outputs[:5]}")
        return outputs


class EnsembleOntologyImplementation(OntologyInterface):
    def __init__(self, ontology_name : str, vlm_names : list[str]):

        vlm_factory = VlmFactory()
        ontology_factory = OntologyFactory()


        self.vlms = [vlm_factory.get_vlm(vlm_name) for vlm_name in vlm_names]
        self.ontology = ontology_factory.get_ontology(ontology_name)
        self.ontology_name = ontology_name
        self.prompt_functions = self.get_prefix_prompt_functions()

        # for key in consts.OMIT_KEYWORDS:
        #     if key in self.ontology: self.ontology.remove(key)

        self.texts = [self.prompt_functions[self.ontology_name](t) for t in self.ontology]
        # self.texts = [f'A photo of {t}' for t in self.ontology]
        len_texts = len(self.texts)
        print(f"Length of ontology: {len_texts}")
        

    def compute_scores(self, image) -> list[(str, float)]:

        outputs = []
        
        for vlm in self.vlms:

            vlm.device = torch.device('cuda')
            vlm.model.to(vlm.device)
            
            temp_outputs = []

            # If VLM crashes, you can extend 10 to bigger number.
            div_texts = len(self.texts) // 10
            len_texts = len(self.texts) 
            for i in range(0, len_texts, div_texts):
                if (i + div_texts) > len_texts:
                    scores = vlm.compute_similarity(image, self.texts[i:i + (len_texts - i)])
                    for j, score in enumerate(scores):
                        temp_outputs.append((self.ontology[i + j], score))
                else:
                    scores = vlm.compute_similarity(image, self.texts[i:i + div_texts])
                    for j, score in enumerate(scores):
                        temp_outputs.append((self.ontology[i + j], score))
            
            outputs.append(temp_outputs)
            # outputs.sort(key=lambda a: a[1], reverse=True)
            # print(f"Top 5: {outputs[:5]}")
            vlm.model.to('cpu')
            vlm.device = torch.device('cpu')
        return outputs
    

class AdhocOntologyImplementation(OntologyInterface):
    def __init__(self, ontology_list : list[str], vlm_name : str):

        vlm_factory = VlmFactory()
        ontology_factory = OntologyFactory()

        self.vlm = vlm_factory.get_vlm(vlm_name)
        self.ontology = ontology_list
        self.ontology_name = ontology_name
        self.prompt_functions = get_prefix_prompt_functions()
        
        # for key in consts.OMIT_KEYWORDS:
        #     if key in self.ontology: self.ontology.remove(key)

        self.texts = [self.prompt_functions[self.ontology_name](t) for t in self.ontology]
        print(f"Length of ontology: {len(self.texts)}")
    

    def compute_scores(self, image) -> list[(str, float)]:
        
        outputs = []

        texts = self.texts

        # If VLM crashes, you can extend 10 to bigger number.
        div_texts = len(texts) // 20
        len_texts = len(texts) 
        for i in range(0, len_texts, div_texts):
            if (i + div_texts) > len_texts:
                scores = self.vlm.compute_similarity(image, texts[i:i + (len_texts - i)])
                for j, score in enumerate(scores):
                    outputs.append((self.ontology[i + j], score))
            else:
                scores = self.vlm.compute_similarity(image, texts[i:i + div_texts])
                for j, score in enumerate(scores):
                    outputs.append((self.ontology[i + j], score))
        
        # outputs.sort(key=lambda a: a[1], reverse=True)
        # print(f"Top 5: {outputs[:5]}")
        return outputs

def main():
    
    ontology_implementation = SingleOntologyImplementation('objects', 'clip')

    image = ontology_implementation.vlm.load_image_url("https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg")
    ontology_implementation.compute_scores(image)

if __name__ == "__main__":
    pass
    # main()


#SingleAdhoc(VLM_NAME, List of words)
# Save the list.