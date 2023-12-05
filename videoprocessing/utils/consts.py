import os
base_path = '/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg'

object_json_path = os.path.join(base_path, 'openimage_classes_all_cleaned_fictional_characters.json')
vg_object_json_path = os.path.join(base_path, 'objects_sorted_all.json')
attribute_json_path = os.path.join(base_path, 'vg_original_attributes_synsets_keys_cleaned_remove_similar0.9.json')
vg_attribute_json_path = os.path.join(base_path, 'attr_sorted_all.json')
scene_json_path = os.path.join(base_path, 'place365_ontology.json')
persons_json_path = os.path.join(base_path, 'persons_ontology.json')
verb_json_path = os.path.join(base_path, 'vg_srl_selected_object_synsets_keys_remove_similar0.9.json')
vg_verb_json_path = os.path.join(base_path, 'capable_of_sorted_all.json')
indoor_json_path = os.path.join(base_path, 'indoor_ontology.json')

OMIT_KEYWORDS = ['media player', 'video', 'playing video', 'audio', 'sound', 'taking video',
                        'water mark', 'water marked', 'watermark', 'watermarks', 'for sale in',
                        'sold from', 'stock', 'sold on', 'by viewers', 'are provided by', 'are posted on',
                        'for more', 'tag with', 'stream from', 'viewed from', 'showing video of', 'are on at',
                        'shuttlecock', 'shutter', 'shutter is white', 'shutters have bones', 'tape is looped',
                        'bliss wants you', 'thumbnail', 'technique'
                ]