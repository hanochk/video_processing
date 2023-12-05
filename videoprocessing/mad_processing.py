import json
import os
import pandas as pd
import numpy as np
import json
import h5py
import pprint


class MADProcessing:
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.train_ann = None
        self.test_ann = None
        self.val_ann = None
        mf = open('/home/paperspace/data/MAD/MAD_id2imdb_names.json')
        self.name_conversion = json.load(mf)
        pass

    def load_annotations(self):
        self.train_ann = json.load(open(os.path.join(self.main_folder, 'annotations/MAD_train.json'), 'r'))
        self.test_ann = json.load(open(os.path.join(self.main_folder, 'annotations/MAD_test.json'), 'r'))
        self.val_ann = json.load(open(os.path.join(self.main_folder, 'annotations/MAD_val.json'), 'r'))

    def get_train_movies(self):
        # Train movies do not have titles, only number. The translation between the numbers and the titles is
        # possible via MAD_id2imdb.json and then https://www.imdb.com/title/IMDB_ID/
        return list(set([v['movie'] for v in self.train_ann.values()]))

    def get_test_movies(self):
        return list(set([v['movie'] for v in self.test_ann.values()]))

    def get_val_movies(self):
        return list(set([v['movie'] for v in self.val_ann.values()]))

    def get_movie_details(self, movie_name):
        for v in self.val_ann.values():
            if movie_name in v['movie']:
                print(v)

    def create_annotation_csv(self, movie_type):
        global_start_list = []
        global_end_list = []
        global_name_list = []
        global_text_list = []
        if movie_type == 'val':
            val_movies = self.get_val_movies()
            value_arr = self.val_ann.values()
        elif movie_type == 'test':
            val_movies = self.get_test_movies()
            value_arr = self.test_ann.values()
        elif movie_type == 'train':
            val_movies = self.get_train_movies()
            value_arr = self.train_ann.values()
        else:
            val_movies = self.get_val_movies()
            value_arr = self.val_ann.values()

        for movie in val_movies:
            start_list = []
            end_list = []
            text_list = []
            name_list = []
            for ann in value_arr:
                if ann['movie'] != movie:
                    continue
                start_list.append(ann['timestamps'][0])
                end_list.append(ann['timestamps'][1])
                text_list.append(ann['sentence'])
                if movie.isnumeric():
                    if movie in self.name_conversion:
                        name_list.append(self.name_conversion[movie])
                    else:
                        name_list.append(movie)
                else:
                    name_list.append(movie)
            ind = np.argsort(np.array(start_list))
            text_list = list((np.array(text_list))[ind])
            start_list = list((np.array(start_list))[ind])
            end_list = list((np.array(end_list))[ind])
            name_list = list((np.array(name_list))[ind])

            global_text_list.extend(text_list)
            global_start_list.extend(start_list)
            global_end_list.extend(end_list)
            global_name_list.extend(name_list)

        df = pd.DataFrame({'name': global_name_list, 'start': global_start_list, 'end': global_end_list,
                           'text': global_text_list})
        df.to_csv('/home/paperspace/data/' + 'movies_' + movie_type + '.csv', index=False)
        pass


def test1():
    mad = MADProcessing('/home/paperspace/data/MAD/')
    mad.load_annotations()
    # mad.get_movie_details('27_Dresses')
    mad.get_movie_details('Lost_Weekend')
    # train_movies = mad.get_train_movies()
    # test_movies = mad.get_test_movies()
    # val_movies = mad.get_val_movies()
    pass

    # Get all the information from 27 Dresses movies (in validation)


import csv
def read_lsmdc(dataset_path):
    new_db = []
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            key = row[0].split()[0]
            name = key[:key.find('.') - 3]
            new_db.append(name)
    return set(new_db)

def test2():
    mad = MADProcessing('/home/paperspace/data/MAD/')
    mad.load_annotations()
    mad.create_annotation_csv('val')
    mad.create_annotation_csv('train')
    mad.create_annotation_csv('test')


if __name__ == "__main__":
    print('Testing MAD dataset')

    test2()

    # # Example of processing a MAD movie
    # name1 = '/home/paperspace/deployment/nebula3_videoprocessing/videoprocessing/dataset1/lsmdc/LSMDC16_annos_test.csv'
    # name2 = '/home/paperspace/deployment/nebula3_videoprocessing/videoprocessing/dataset1/lsmdc/LSMDC16_annos_training.csv'
    # name3 = '/home/paperspace/deployment/nebula3_videoprocessing/videoprocessing/dataset1/lsmdc/LSMDC16_annos_val.csv'
    #
    # db1 = read_lsmdc(name1)
    # db2 = read_lsmdc(name2)
    # db3 = read_lsmdc(name3)
    # test1()