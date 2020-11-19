import pickle as pkl
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt


class Manager:
    def __init__(self, dir: str) -> None:
        self.dir = dir

    def load_file(self, file_name: str) -> None:
        with open(f'{self.dir}/{file_name}.pkl', 'rb') as f:
            self.data = pkl.load(f)

    def save_file(self, file_name: str) -> None:
        with open(f'{self.dir}/{file_name}.pkl', 'wb') as f:
            pkl.dump(self.data, f, pkl.HIGHEST_PROTOCOL)

    def rename(self, name: str, new_name: str) -> None:
        if name in self.data:
            self.data[new_name] = self.data.pop(name)
        else:
            print(f'{name} not found')

    def merge(self, names: Iterable[str]) -> None:
        for name in names:
            if name not in self.data:
                print(f'{name} not found')
                return

        merged_dict = self.data[names[0]]
        for name in names[1:]:
            n_dict = self.data.pop(name)

            paths = n_dict['paths']
            merged_dict['paths'].update(paths)

    def remove(self, names: Iterable[str]) -> None:
        for name in names:
            if name not in self.data:
                print(f'{name} not found')
                return

        paths_l = []
        for name in names:
            n_paths = self.data.pop(name)['paths']
            paths_l.append(n_paths)

        for paths in paths_l:
            for person_dict in self.data.values():
                paths -= person_dict['paths']
            self.data['none']['paths'].update(paths)

    def show(self) -> None:
        names = []
        faces = []
        for name, person_dict in self.data.items():
            if name != 'none':
                names.append(name)
                faces.append(person_dict['face'])

        face_cat = np.concatenate(faces, axis=0)
        plt.imshow(face_cat)
        plt.yticks(np.arange(80, len(names) * 160, 160), labels=names)
        plt.xticks([])
        plt.show()
