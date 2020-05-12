import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
import os
from multiprocessing import Process, Queue, current_process, cpu_count, freeze_support
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import time
import random
import cv2
import string
#from imageai.Prediction.Custom import CustomImagePrediction

def read_labels_and_images():
    start_time = time.time()
    path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_train"

    #get file list
    file_list = find_files(path)

    #return a list of train files, train labels, prediction files, and prediction labels
    print("Grab the files")
    train_imgs, train_labels, predict_imgs, predict_labels = [], [], [], [] #holders

    processed = process_files(file_list)
    train_imgs += processed[0]
    train_labels += processed[1]
    predict_imgs += processed[2]
    predict_labels += processed[3]

    print("Read and processed images. Time =", time.time()-start_time)

    return(train_imgs,train_labels,predict_imgs,predict_labels)
    #return(process_files(file_list))


#get all the files in the training directory
def find_files(path):
    alpha = list(string.ascii_uppercase)
    #find all the files, add to list
    #get a list of all folders 
    folderlist = [f.path for f in os.scandir(path) if f.is_dir() and os.path.basename(f.path) in alpha]
    filelist = []
    for f in folderlist:
        #add the files in the folder list to the filelist
        #   as tuples of (path,name)
        #use below for only a portion of the data
        filelist.extend([fil.path for fil in os.scandir(f) if ((fil.path.find("jpg") != -1) and (random.random()<0.01))])
        #Note: os.path.dirname(path) gives parent directory


    print("Found "+ str(len(filelist)) + " files")

    return filelist

#Convert images to tensors and map them to their labels, split training/prediction
def process_files(files):
    t, tl, p, pl = [], [], [], [] #train, train_labels, predict, predict_labels
    checkpoint, check_interval, num_done = 0, 5, 0 # Just for showing progress

    alpha = list(string.ascii_uppercase)
    alpha_dict = {alpha[n]:n for n in range(26)}

    for filename in files:
        if 100*num_done > (checkpoint + check_interval) * len(files):
            checkpoint += check_interval
            print((int)(100 * num_done / len(files)), "% done")
        num_done += 1
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = tf.reshape(img, [200, 200, 1])
        if np.random.random() < 0.9: # subset
            t.append(img.numpy()) 
            tl.append(alpha_dict[os.path.basename(os.path.dirname(filename))])
        else:
            #predict these
            p.append(img.numpy())
            #print("img")
            #print(img)
            #print("img.numpy")
            #print(img.numpy())
            pl.append(alpha_dict[os.path.basename(os.path.dirname(filename))])
        
    return (t,tl,p,pl)

def build_letter_model():
    # Build the model

    
    #hochbergs options with some mods
    model = tf.keras.Sequential([                                                   #sequential Model
        tf.keras.layers.Conv2D(8, (7,7), padding='same', activation='relu'),        #2D Conv., increasing layer output
        tf.keras.layers.MaxPooling2D((2, 2)),                                       #Kernal size 7->5->3 VERY important
        tf.keras.layers.Dropout(rate=0.5),                                          #2,2 Pooling seemed to work fine
        tf.keras.layers.Conv2D(24, (5,5), padding='same', activation='relu'),       #0.5 Dropout worked fine
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),                                   #reduce all layers to depth
        tf.keras.layers.Dense(26, activation='softmax')])                           #good old soft max reduction with 26 outputs
    print("Shape of output", model.compute_output_shape(input_shape=(None, 200, 200, 1)))


    model.compile(optimizer=tf.keras.optimizers.Adam(),                             #Adam performed best after small tests
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),               #loss function for non-onehot, >2 outputs
                metrics=['accuracy'])
    

    '''
    #Google keras model for classification on images
    model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1")],
    tf.keras.layers.Dense(26,activation="softmax"))

    #model compile failing on all tested formats, find another

    '''

    '''
    #hochbergs trainer as given
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4",
                       trainable=False),  
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    model.build([None, 200, 200, 26])  # Batch input shape.
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    '''
    

    #model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    print("Done building the network topology.")

    # Read training and prediction data
    #path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_train/processed_sm" #1%
    #path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_train/processed_med" #10%
    path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_train/processed" #Full Set
    train_path = path + "/train.npy"
    train = np.load(train_path)
    train_labels_path = path + "/train_labels.npy"
    train_labels = np.load(train_labels_path)
    predict_path = path + "/predict.npy"
    predict = np.load(predict_path)
    predict_labels_path = path + "/predict_labels.npy"
    predict_labels = np.load(predict_labels_path)


    #train, train_labels, predict, predict_labels = read_labels_and_images()
    #print(train)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(path + "/models/steps/checkpoint_{epoch}")

    # Train the network
    print("Starting to train the network, on", len(train), "samples.")

    path = "C:/ASL Research Images/asl-alphabet"

    model.fit(np.asarray(train, dtype=np.float32), np.asarray(train_labels), epochs=20, batch_size=64, verbose=2, callbacks=[checkpoint]) # failing on invalid path in keras
    model.save(path + "/models")
    print("Done training network.")

    # Predict
    p = model.predict_classes(np.asarray(predict,np.float32))
    num_correct, num_total = 0, 0
    for pr, pl in zip(p, predict_labels):
        print("Predict", pr, "\tActual", pl, "***" if (pr != pl) else ".")
        if pr == pl: num_correct += 1
        num_total += 1
    print("Accuracy on prediction data", (100 * num_correct)/num_total)
        
    return model

#Create a preprocessed file of the data to speed up training
def make_transformed_copy():
    path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_train/processed"
    train, train_labels, predict, predict_labels = read_labels_and_images()
    #save train
    train_path = path + "/train.npy"
    np.save(train_path,np.asarray(train, dtype=np.float32))
    print("saved train")
    #save train_labels
    train_labels_path = path + "/train_labels.npy"
    np.save(train_labels_path,np.asarray(train_labels, dtype=np.float32))
    print("saved train labels")
    #save predict
    predict_path = path + "/predict.npy"
    np.save(predict_path,np.asarray(predict,np.float32))

    print("saved predict")
    #save predict_labels
    predict_labels_path = path + "/predict_labels.npy"
    np.save(predict_labels_path,np.asarray(predict_labels,np.float32))
    print("saved predict labels")

def image_ai_test():
    #tomorrow
    return
        
def load_and_visualize_model(filename):
    np.set_printoptions(precision=3, suppress=True)
    model = tf.keras.models.load_model(filename)
    v = tf.squeeze(model.trainable_variables[0]).numpy()
    num_features = v.shape[2]
    fig, ax = plt.subplots(nrows=4, ncols=num_features//4)
    ax = ax.flatten()
    for i in range(num_features):
        img = v[:,:,i]
        ax[i].imshow(img, cmap='Greys')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        print(img)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename + ".png")



# Reads model saved in filename on disk, runs it using webcam input
def load_and_run_model():
    alpha = list(string.ascii_uppercase)
    model = tf.keras.models.load_model("C:/ASL Research Images/asl-alphabet/models")
    cap = cv2.VideoCapture(0)
    keep_going = True
    while keep_going:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[:,(w-h)//2:(w+h//2)]
        crop = cv2.resize(crop, (100,100))
        tf_img = tf.reshape(crop, [100,100, 1])

        p = model.predict_classes(np.asarray([tf_img], dtype=np.float32), batch_size=1)[0]
        print(p)
        #p = model.predict(np.asarray([tf_img], dtype=np.float32))[0] # See the raw data
        
        color = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR),(256, 256))
        cv2.putText(color, alpha[p+1], (20, 30) , cv2.FONT_HERSHEY_SIMPLEX,\
                        1.0,(255,0, 0),2, lineType=cv2.LINE_AA)
        cv2.imshow("ASL Recognizer", color)
        input("Press Enter to continue...")
        if cv2.waitKey(100) & 0xFF == ord(' '):
            keep_going = False

    cap.release()
    cv2.destroyAllWindows()

def show_on_test_data():
    alpha = list(string.ascii_uppercase)
    model = tf.keras.models.load_model("C:/ASL Research Images/asl-alphabet/models")
    path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_test"
    imgs = [get_img(fil.path) for fil in os.scandir(path) if ((fil.path.find("jpg") != -1))]
    for img in imgs:
        tf_img = tf.reshape(img, [200,200,1])
        p = model.predict_classes(np.asarray([img], dtype=np.float32), batch_size=1,verbose=1)[0]
        cv2.putText(img, alpha[p+1], (20, 30) , cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,0, 0),2, lineType=cv2.LINE_AA)
        cv2.imshow(img)


def get_img(f):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(gray,(200,200))
    return crop


def capture_custom_test():
    #test()
    save()

def test():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imshow("ASL Recognizer", frame)
    ret, frame = cap.read()
    keep_going = True
    ret, frame = cap.read()
    while(keep_going):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dh = 120
        crop = gray[dh//2:h-dh//2,(w-(h-dh))//2:w-((w-(h-dh))//2)]
        #crop = cv2.resize(crop, (200,200))
        cv2.imshow("ASL Recognizer", crop)
        if cv2.waitKey(100) & 0xFF == ord(' '):
            keep_going = False

def save():
    alpha = list(string.ascii_uppercase)
    cap = cv2.VideoCapture(0)
    model = tf.keras.models.load_model("C:/ASL Research Images/asl-alphabet/models")
    for i in range(3):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[:,(w-h)//2:(w+h//2)]
        crop = cv2.resize(crop, (100,100))
        color = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR),(256, 256))

        tf_img = tf.reshape(crop, [100,100, 1])

        p = model.predict_classes(np.asarray([tf_img], dtype=np.float32), batch_size=1)[0]
        print(p)
        cv2.putText(color, alpha[p+1], (20, 30) , cv2.FONT_HERSHEY_SIMPLEX,\
                        1.0,(255,0, 0),2, lineType=cv2.LINE_AA)
        #p = model.predict(np.asarray([tf_img], dtype=np.float32))[0] # See the raw data
        cv2.imshow("ASL Recognizer", color)
        input(str(3-i) + " tests remaining. Press Enter to continue...")

def test_model():
    alpha = list(string.ascii_uppercase)
    path = "C:/ASL Research Images/asl-alphabet/asl_alphabet_train/processed" #Full Set
    predict_path = path + "/predict.npy"
    predict = np.load(predict_path)
    predict_labels_path = path + "/predict_labels.npy"
    predict_labels = np.load(predict_labels_path)
    train_labels = np.load(path + "/train_labels.npy")
    model = tf.keras.models.load_model("C:/ASL Research Images/asl-alphabet/models")
    p = model.predict_classes(np.asarray(predict,np.float32))
    num_correct, num_total = 0, 0
    print(uniquesc(predict_labels))
    lists = [[] for i in range(25)]
    for pr, pl in zip(p, predict_labels):
        print("Predict", pr, "\tActual", pl, "***" if (pr != pl) else ".")
        lists[int(pl)].append(int(pr))
        if pr == pl: num_correct += 1
        num_total += 1
    count = 0

    for l in lists:
        if len(l)>0:
            total = len(l)
            a = 0
            for n in l:
                if int(n)==count:
                    a+=1
            print(alpha[count] + "accuracy is " + str(a/total))
        count+=1

def uniquesc(lw):
    uniques = []
    alpha = list(string.ascii_uppercase)
    for w in lw:
        if w not in uniques:
            uniques.append(w)
    uniques = [alpha[int(n)] for n in uniques]
    
    return uniques



#make_transformed_copy()
#build_letter_model()
#make_transformed_copy()
#load_and_visualize_model("C:/ASL Research Images/asl-alphabet/models")
#load_and_run_model()
#capture_custom_test()
#show_on_test_data()
test_model()
