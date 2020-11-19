from matplotlib import pyplot as plt

import config
from scanner import Scanner

if __name__ == "__main__":
    sc = Scanner()

    sc.scan(
        search_dir=config.BASE_DIR,
        extensions=config.IMG_EXT,
        detection_threshold=config.DETECTION_THRESHOLD,
        recognition_threshold=config.RECOGNITION_THRESHOLD,
        img_resize_width=config.MAX_IMG_WIDTH,
        img_resize_height=config.MAX_IMG_HEIGHT,
        verbose=True,
    )

    for person, person_dict in sc.people.items():
        print(f'Person: {person}')
        for img in person_dict['paths']:
            print(img)
        if person != 'none':
            face = person_dict['face']
            plt.imshow(face)
            plt.show()
        print('')
