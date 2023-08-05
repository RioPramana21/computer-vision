import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    name_list = os.listdir(root_path)
    return name_list

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    img_list = []
    class_list = []

    for index, name in enumerate(train_names):
        image_path = f"{root_path}/{name}"
        for img in os.listdir(image_path): #ambil semua image yang ada di folder sekarang
            full_image_path = f"{image_path}/{img}" #ambil path untuk diambil imagenya
            img = cv2.imread(full_image_path) #baca imagenya
            # Resize image (hanya dilakukan untuk training dataset)
            # Dari berbagai eksperimen yang saya lakukan, dimension (300,350) memberikan..
            #.. hasil yang paling baik
            img = cv2.resize(img, (300, 350), interpolation = cv2.INTER_CUBIC)
            img_list.append(img) #simpan imagenya
            class_list.append(index) #simpan classnya

    return img_list, class_list
 

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    # Untuk membedakan dataset apa yang sedang menggunakan function ini:
    isTrainDataset = True
    if (image_classes_list == None):
        isTrainDataset = False
    
    # List yang akan direturn
    cropped_gray_face_list = []
    face_location_rect_list = []
    filtered_class_list = []

    # buat face classifiernya menggunakan .xml dari haarcascade
    haarcascade_path = "./haarcascades/haarcascades/haarcascade_frontalface_alt2.xml"
    face_clf = cv2.CascadeClassifier(haarcascade_path)
    
    for index, img in enumerate(image_list):
        # Sebelum deteksi, ubah dulu ke gray image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Deteksi faces yang ada di image sekarang
        detected_faces = face_clf.detectMultiScale(
            image = gray_img,
            scaleFactor = 1.15, #mengecilkan 15% setiap loop
            minNeighbors = 4,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        ) #return jumlah face yg ke detect di 1 image tersebut

        # Filter image:
        num_of_detected_face = len(detected_faces)
        if (num_of_detected_face != 1):
            continue 

        for face_rect in detected_faces:
            face_location_rect_list.append(face_rect) #simpan rectangle lokasi dan size facenya
            x, y, w, h = face_rect
            face_img = gray_img[y:y+h, x:x+w] # crop imagenya sesuai face rectanglenya
            cropped_gray_face_list.append(face_img) #simpan hasil cropnya
            if (isTrainDataset):
                # Jika yang menggunakan function ini adalah train dataset..
                #..simpan classnya
                filtered_class_list.append(image_classes_list[index])
    
    return cropped_gray_face_list, face_location_rect_list, filtered_class_list


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    img_list = []
    for image_path in os.listdir(test_root_path):
        full_image_path = f"{test_root_path}/{image_path}"
        img = cv2.imread(full_image_path)
        img_list.append(img)
    
    return img_list
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    results = []
    for face_img in test_faces_gray:
        result, _ = recognizer.predict(face_img)
        results.append(result)
    return results

def get_wanted_status(prediction_result, train_names, wanted_names):
    '''
        To generate a list of wanted status (wanted or safe) from prediction results

        Parameters
        ----------
        prediction_result : list
            List containing all wanted results from given test faces
        test_image_list : list
            List containing all loaded test images
        wanted_names : list
            List containing all wanted names
        
        Returns
        -------
        list
            List containing all verification status from prediction results
    '''
    verifications = []
    for prediction in prediction_result:
        if (train_names[prediction] in wanted_names):
            # Simpan juga hasil prediksinya agar bisa digunakan di function berikutnya
            verifications.append((prediction, "Wanted"))
        else:
            verifications.append((prediction, "Safe"))
    
    return verifications

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    final_images = []
    for index, img in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[index]
        # Ambil hasil prediction dan nama verifikasinya untuk display
        predicted_class = predict_results[index][0]
        verification_status = predict_results[index][1]
        # Set color hijau/merah
        color = (0,255,0)
        if (verification_status == "Wanted"):
            color = (0,0,255)
        #Gambar rectangle untuk facenya
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 10)
        # Tentukan font scale agar menyesuaikan dengan size image
        f_scale = 0.004
        font_scale = min(w, h) * f_scale
        # Taruh text untuk menunjukkan nama classnya
        text = f"{train_names[predicted_class]}"
        cv2.putText(img, text, (x-25, y-10), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
        # Taruh text untuk menunjukkan status verifikasinya
        text = f"{verification_status}"
        cv2.putText(img, text, (x-10, int(y+(1.25 * h))), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 4)

        final_images.append((verification_status, img))
    
    return final_images
    

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    sorted_img_list = []
    # Ambil images yang Wanted terlebih dahulu
    for img in image_list:
        if (img[0] == "Wanted"):
            wanted_img = cv2.cvtColor(img[1], cv2.COLOR_BGR2RGB)
            wanted_img = cv2.resize(wanted_img, (255, 330), interpolation = cv2.INTER_AREA)
            sorted_img_list.append(wanted_img)
    # Kemudian ambil yang Safe
    for img in image_list:
        if (img[0] != "Wanted"):
            safe_img = cv2.cvtColor(img[1], cv2.COLOR_BGR2RGB)
            safe_img = cv2.resize(safe_img, (255, 330), interpolation = cv2.INTER_AREA)
            sorted_img_list.append(safe_img)

    # Display semuanya dalam 1 figure
    plt.figure(figsize=(10, 8))
    for index, img in enumerate(sorted_img_list):
        plt.subplot(2, 3, index + 1)
        plt.imshow(img, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "./dataset/train"

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)

    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)

    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "./dataset/test"
    wanted_names = [
        "Jackie Chan",
        "Cho Yi-Hyun",
        "Kim Se-jeong"
        ]

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)

    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)

    predict_results = predict(recognizer, test_faces_gray)

    verification_statuses = get_wanted_status(predict_results, train_names, wanted_names)

    predicted_test_image_list = draw_prediction_results(verification_statuses, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list)