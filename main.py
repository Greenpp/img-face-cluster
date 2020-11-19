import config
from manager import Manager
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

    mng = Manager(config.BASE_DIR)
    mng.load_file('face-recognition-results')
    mng.show()
