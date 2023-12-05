from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForNextSentencePrediction
import numpy as np
import random
import sys
sys.path.insert(0, './')
sys.path.insert(0, 'nebula3_database/')
from nebula3_database.playground_db import PLAYGROUND_DB

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
device = "cpu"
model = model.to(device)

def sentence_compatability(sent_a, sent_b):

    encoded = tokenizer.encode_plus(sent_a, text_pair=sent_b, return_tensors='pt').to(device)
    seq_relationship_logits = model(**encoded)[0]
    probs = softmax(seq_relationship_logits, dim=1)
    score = probs[0][0].tolist()
    # print(score)
            
    return score

def scene_element_compatability(scene_sentences):
    scores = []
    for i in range(len(scene_sentences) - 1):
        cur_sen = scene_sentences[i]
        next_sen = scene_sentences[i+1]
        score = sentence_compatability(cur_sen, next_sen)
        scores.append(score)
    mul_scores = np.prod(scores)
    if len(scene_sentences) == 1:
        norm_scores = 1.0
    else:
        norm_scores = np.power(mul_scores, 1 / len(scene_sentences))
    # print(f"Scores: {[scores]}")
    # print(f"Normalized score: {norm_scores}")
    return norm_scores

def document_compatability(doc_name):
    db_instance = PLAYGROUND_DB()
    doc = db_instance.get_document(doc_name)

    movies_dict = {}
    movies_dict_sorted = {}
    # Get all the movies and its groundtruth & index
    for element in doc:
        File = element['File']
        groundtruth = element['groundtruth'][0]
        index = element['index']
        if File not in movies_dict:
            movies_dict[File] = {} 
            movies_dict[File][index] = groundtruth
        else:
            movies_dict[File][index] = groundtruth

    # Sort all the movies by index
    for movie in movies_dict:
        movies_dict_sorted[movie] = sorted(movies_dict[movie].items(), key=lambda item: item[0])
    
    # Calculate scores for every movie
    scores = []
    for movie in movies_dict_sorted:
        sentences = []
        # If we have at least two groundtruths
        if len(movies_dict_sorted[movie]) > 1:
            for sentence in movies_dict_sorted[movie]:
                sentences.append(sentence[1])
            score = scene_element_compatability(sentences)
            scores.append(score)
    scores_average = np.mean(scores)
    print("\n##### SCORES on ALL S2_CLSMDC movies #####")
    print(f"Scores: {scores}")
    print(f"Mean Scores: {scores_average}\n")

def random_document_compatability(doc_name='s2_clsmdc', num_rand_movies = 20):
    db_instance = PLAYGROUND_DB()
    doc = db_instance.get_document(doc_name)

    sentences = []
    # Get all the groundtruth sentences
    for element in doc:
        sentence = element['groundtruth'][0]
        sentences.append(sentence)

    random.shuffle(sentences)

    scores = []
    for i in range(num_rand_movies):
        random_sentences = sentences[0:random.randint(5, 12)]
        random.shuffle(sentences)
        score = scene_element_compatability(random_sentences)
        scores.append(score)
    scores_average = np.mean(scores)
    print("##### SCORES on RANDOM movies #####")
    print(f"Scores: {scores}")
    print(f"Average scores: {scores_average}")


    

def main():
    
    document_compatability(doc_name='s2_clsmdc')
    random_document_compatability(doc_name='s2_clsmdc', num_rand_movies=20)
    sent_a = "An old man with a button t-shirt lying down in the bath filled with water and his eyes are closed."
    sent_b = "A blonde woman in a blue dress is standing in a room next to the door."
    score = sentence_compatability(sent_a, sent_b)
    print("\n #### JUST TWO SENTENCES of the same scene element ####")
    print(f"Score of two sentences: {score}")
    
    # Test on movie: 3006_A_GOOD_DAY_TO_DIE_HARD_01.30.26.150-01.30.41.504
    scene_sentences = ["a pilot is sitting in the cockpit of a white airplane, and there are poles in the background",
                        "the sun is shining in the background while a man is sitting in the cockpit of an airplane in the airport",
                        "A soldier holding a gun, wearing a hat and sun glasses, in the background soldiers stand next to cars",
                        "Two soldiers holding guns, wearing vest and uniform, stand near a truck",
                        "Two soldiers with guns standing in front of cars",
                        "An airport filled with cars and a couple of people standing in front of them.",
                        "An airplane has arrived at the airport, while a man is talking on the cellphone",
                        "A soldier with a weapon and a woman with a bag are standing in front of a truck."]
    print("\n#### SCENE ELEMENT SENTENCES from s2_clsmdc: ####")
    scene_element_compatability(scene_sentences)

    # Test on movie: 3049_MORNING_GLORY_01.40.02.750-01.40.17.024
    scene_sentences = ["a woman walking to the kitchen watching an old man preparing food in a kitchen, a video camera on the right",
                        "a woman in a brown dress standing and watching something in a living room, a camera on the left , a lining room on the back",
                        "an old man cooking in the kitchen, staring at someone, wearing a white shirt and a yellow tie",
                        "A beautiful woman wearing a white dress, smiling and looking over old man wearing a white shirt and a yellow tie cooking in the kitchen",
                        "a woman with brown hair and a red lipstick smiles at the camera",
                        "an older man wearing a white shirt and a yellow tie, gazing at the camera"]
    print("\n#### SCENE ELEMENT SENTENCES from s2_clsmdc: ####")
    scene_element_compatability(scene_sentences)

    # Test on movie: 3089_XMEN_FIRST_CLASS_01.54.31.621-01.54.51.029
    scene_sentences = ["An military man with a helmet and a vest stands concentrated",
                        "A large navy destroyer sails and shoots its two cannons while other ships are in the background in a battle at sea",
                        "Two battries of cannons on a battleship with ladders and iron turrets",
                        "A group of battleships in the distant water with smoke coming out of them",
                        "Missiles going up vertically into the sky",
                        "A scene from a scifi movie where two teenagers and two aliens with dark faces are standing, watching in concentration, dressed with black and yellow swim vests",
                        "A young man with a long dark hair and a suit, a lady with long dark hair and earings and a red-faced bearded alien standing and watching in concentration ",
                        "A group of missiles flying in the sky towards a mountain and over beaches and sea leaving a trail of smoke behind",
                        "A view from above of number of ships sailing and launching a large number of missiles",
                        "The missiles are flying over a sea towards a round bay with and a green land. The missiles fly perpendicularly to the ships route",
                        "Four persons in yellow and black wetsuits shown from behind, are looking alerted at the sea and the sky where a group of missiles are flying towards them",
                        "A group of missiles are flying over the sea towards a beach under a green mountain leaving a trail of smoke behind",
                        "A movie scene of a man with a yellow wetsuit, black helmet nad blue eyes in front of other two men with yellow and black wetsuits",
                        "A large number of missiles flying over the ocean towards teh camera, leaving a white trail of smoke",
                        "missiles have four small wings"]
    print("\n#### SCENE ELEMENT SENTENCES from s2_clsmdc: ####")
    scene_element_compatability(scene_sentences)

    random_sentences = ["Asian men standing in a crowd and watching.",
                        "An image of a back of a bald man at night.",
                        "A young boy is performing kong fu in front of a crowd in the stadium.",
                        "A man is falling out of the building.",
                        "a woman walking in a caffe in first floor of big building",
                        "a man sitting in a car in the dark",
                        "a woman standing in the desert with a herd of horses",
                        "two women in a yoga class doing exercises on a blue mat",
                        "A group of battleships in the distant water with smoke coming out of them",
                        "A scene from a scifi movie where two teenagers and two aliens with dark faces are standing, watching in concentration, dressed with black and yellow swim vests"]
    print("\n#### RANDOM SENTENCES from s2_clsmdc: ####")                    
    scene_element_compatability(random_sentences)

    random_sentences = ["A man shoots himself in the leg with a pistol in front of other men in a street",
                        "Marry Jane walks out the building and gets close to fewpeople that are clapping hands",
                        "A man in a coat, a hat and a bow tie stands at the entrance door of house",
                        "A partisan throws an axe to another man who is working with a hammer over a wooden workbench",
                        "A black man in plaid shirt is running within a crowd, carrying a scared girl in his arms",
                        "A group of nazi soldiers are riding on the back of a military truck",
                        "A woman sitting at a dinner table picks up a cup, and cleans the rug",
                        "A man stands in a porch next to a chair and turns on the light",
                        "speed boats cruising fast in a water strip next to a sandy beach and a hotel",
                        "A lady with a white cap walks leisurely between trees in a meadow",
                        "A man with a black jacket and white t-shirt is sitting in a room with a woman that wears a black dress.",
                        "A man in a suit and a tie talks to a blonde lady with earings and winks at her while another man with a moustache laughs in the background"]
    print("\n#### RANDOM SENTENCES from s1_lsmdc: ####")                    
    scene_element_compatability(random_sentences)

if __name__ == "__main__":
    main()