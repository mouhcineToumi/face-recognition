from detector import *
import cv2
import matplotlib.pyplot as plt


class Preprocessor:
    
    def __init__(self, dir_="images/"):
        self.face_detector = FaceDetector()
        self.img_dir = dir_

    #horizontal flip
    def flip_img(self, image):
        return cv2.flip(image, 1)

    #resize
    def resize(self, image, size=(96, 96)):
        return cv2.resize(image, size)


    #takes image and returns two faces
    def process(self, image_path):
        faces = []
        faces.append(self.face_detector.get_face(image_path))
        faces.append(self.face_detector.get_face2(self.rotate_img(image_path, -15)))
        faces.append(self.face_detector.get_face2(self.rotate_img(image_path, 15)))
        d = []
        for f in faces:
            d.append(self.resize(f))
        return d


    def rotate_img(self, path, ang=-20):
        img = cv2.imread(path, 1)
        num_rows, num_cols = img.shape[:2]
        
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), ang, 1)
        d = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))        
        return d


    def get_input(self, image_path):
        face = self.face_detector.get_face(image_path)
        return self.resize(face)
    
    #in a folder containing ( images only )
    # dic = { "person":[img1], ...}
    def get_augmented(self, pdir):
        dic = []
        pics = [pic for pic in os.listdir(self.img_dir + pdir)]
        for pic in pics:
            dic += (self.process(self.img_dir + pdir + pic))
        return dic
    
    def get_database(self):
        db = {}
        people = [person for person in os.listdir(self.img_dir)]
        for person in people:
            db[person] = self.get_augmented(person+"/")
        return db