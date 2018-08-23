import six.moves.cPickle as Pickle
import os


dataset_dir = '/home/suka/dataset/mango/'
clothes = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        clothes.append(filename)

print(len(clothes))


with open('mango.pkl', 'wb') as cloth_table:
    Pickle.dump(clothes, cloth_table)

print('done')