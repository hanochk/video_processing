from nebula3_videoprocessing.videoprocessing.vlm_implementation import ClipVlmImplementation, BlipItcVlmImplementation, BlipItmVlmImplementation, VisualGroundingToVlmAdapter
from nebula3_videoprocessing.videoprocessing.utils.singleton import Singleton
# from nebula3_experts_vg.vg.vg_expert import VisualGroundingVlmImplementation
class VlmFactory:
    _creators = {}
    def __init__(self, metaclass=Singleton): 
        self.vlm_map = {
            'clip': ClipVlmImplementation,
            'blip_itc': BlipItcVlmImplementation,
            'blip_itm': BlipItmVlmImplementation,
            'owl_vit': VisualGroundingToVlmAdapter
            # 'vg': VisualGroundingVlmImplementation
        }

    def register_vlm(self, vlm_name):
        try:
            vlm_implementation = self.vlm_map[vlm_name]
        except:
                dict_keys = self.vlm_map.keys()
                raise Exception("VLM not found. please use on of these keys: {}".format(dict_keys))    

        self._creators[vlm_name] = vlm_implementation()

    def get_vlm(self, vlm_name):
        creator = self._creators.get(vlm_name)
        if not creator:
            try:
                self.register_vlm(vlm_name)
                creator = self._creators.get(vlm_name)
            except:
                dict_keys = self.vlm_map.keys()
                raise Exception("VLM not found. please use on of these keys: {}".format(dict_keys))

        return creator

# def main():
#     vlm1 = VlmFactory().get_vlm("clip")
#     print(vlm1)
#     main1()

# def main1():
#     vlm2 = VlmFactory().get_vlm("clip")
#     print(vlm2)
    
# if __name__ == "__main__":
#     main()