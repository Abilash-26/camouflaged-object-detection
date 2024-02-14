from flask import Flask, render_template,request,make_response
import mysql.connector
from mysql.connector import Error
import sys
import os
import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
from skimage import measure #scikit-learn==0.23.0  scikit-image==0.14.2
#from skimage.measure import structural_similarity as ssim #old
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skmetrics import matrix
import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import json
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import random



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def index1():
    return render_template('index.html')

@app.route('/twoform')
def twoform():
    return render_template('twoform.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')



@app.route('/regdata', methods =  ['GET','POST'])
def regdata():
    connection = mysql.connector.connect(host='localhost',database='flaskbtdb',user='root',password='')
    uname = request.args['uname']
    email = request.args['email']
    phn = request.args['phone']
    pssword = request.args['pswd']
    addr = request.args['addr']
    dob = request.args['dob']
    print(dob)
        
    cursor = connection.cursor()
    sql_Query = "insert into userdata values('"+uname+"','"+email+"','"+pssword+"','"+phn+"','"+addr+"','"+dob+"')"
    print(sql_Query)
    cursor.execute(sql_Query)
    connection.commit() 
    connection.close()
    cursor.close()
    msg="User Account Created Successfully"    
    resp = make_response(json.dumps(msg))
    return resp


def mse(imageA, imageB):    
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    print(imageA)
    #s = ssim(imageA, imageB) #old
    s = measure.compare_ssim(imageA, imageB, multichannel=True)
    return s



"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
    connection=mysql.connector.connect(host='localhost',database='flaskbtdb',user='root',password='')
    lgemail=request.args['email']
    lgpssword=request.args['password']
    print(lgemail, flush=True)
    print(lgpssword, flush=True)
    cursor = connection.cursor()
    sq_query="select count(*) from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)
    
    connection.commit() 
    connection.close()
    cursor.close()
    
    if rcount>0:
        msg="Success"
        resp = make_response(json.dumps(msg))
        return resp
    else:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp
        

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
    
        prod_mas = request.files['first_image']
        print(prod_mas)
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join("E:\\Upload\\", filename))

        #csv reader
        fn = os.path.join("E:\\Upload\\", filename)

        count = 0
        imagelist=os.listdir('static/Dataset')
        print(imagelist)
        width = 400
        height = 400
        dim = (width, height)
        ci=cv2.imread("E:\\Upload\\"+ filename)
        gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/Grayscale/"+filename,gray)

        #cv2.imshow("org",gray)
        #cv2.waitKey()

        thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        cv2.imwrite("static/Threshold/"+filename,thresh)
        #cv2.imshow("org",thresh)
        #cv2.waitKey()

        lower_green = np.array([34, 177, 76])
        upper_green = np.array([255, 255, 255])
        hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        binary = cv2.inRange(hsv_img, lower_green, upper_green)
        print(binary)
        cv2.imwrite("static/Binary/"+filename,gray)

        indval=0
        filesv = glob.glob('static/data_cleaned/test/*')
        for file in filesv:
            # resize image
            print(file)
            print(filename)
            if filename in file:
                ind=indval
                break
            else:
                indval=indval+1
        
        #ind=0

        '''
        flagger=1
        imagename=""
        oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
        for i in range(len(imagelist)):
            if flagger==1:
                files = glob.glob('static/Dataset/'+imagelist[i]+'/*')
                #print(len(files))
                for file in files:
                    # resize image
                    print(file)
                    oi=cv2.imread(file)
                    resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
                    #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow("comp",oresized)
                    #cv2.waitKey()
                    #cv2.imshow("org",resized)
                    #cv2.waitKey()
                    #ssim_score = structural_similarity(oresized, resized, multichannel=True)
                    #print(ssim_score)
                    ssimscore=compare_images(oresized, resized, "Comparison")
                    if ssimscore>0.6:
                        imagename=imagelist[i]
                        flagger=0
                        break
        '''
        imagename="Detected"
        accuracy=matrix.accuracy()
        filenamer=filename.split('.')
        filenamer=filenamer[0]
        msg=imagename+","+filename+","+str(accuracy)+","+filenamer
        resp = make_response(json.dumps(msg))
        return resp



   

def load_data_set(filename):
    try:
        with open(filename, newline='') as iris:
            return list(reader(iris, delimiter=','))
    except FileNotFoundError as e:
        raise e


def convert_to_float(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([float(x) for x in data[:len(data)-1]] + [data[len(data)-1]])

        elif mode == 'test':
            for data in data_set:
                new_set.append([float(x) for x in data])

        else:
            print('Invalid mode, program will exit.')
            exit()

        return new_set

    except ValueError as v:
        print(v)
        print('Invalid data set format, program will exit.')
        exit()


def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))


def knn(training_set, test_set, k):
    distances = []
    dist = 0
    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)

    try:
        for test_instance in test_set:
            for row in training_set:
                for x, y in zip(row[:limit], test_instance):
                    dist += (x-y) * (x-y)
                distances.append(row + [sqrt(dist)])
                dist = 0

            distances.sort(key=itemgetter(len(distances[0])-1))

            # find k nearest neighbors
            neighbors = find_neighbors(distances, k)

            # get the class with maximum votes
            index, value = find_response(neighbors, classes)

            # Display prediction
            print('The predicted class for sample ' + str(test_instance) + ' is : ' + classes[index])
            print('Number of votes : ' + str(value) + ' out of ' + str(k))

            # empty the distance list
            distances.clear()

    except Exception as e:
        print(e)


def mainknn():
    try:
        # get value of k
        k = int(input('Enter the value of k : '))

        # load the training and test data set
        training_file = input('Enter name of training data file : ')
        test_file = input('Enter name of test data file : ')
        training_set = convert_to_float(load_data_set(training_file), 'training')
        test_set = convert_to_float(load_data_set(test_file), 'test')

        if not training_set:
            print('Empty training set')

        elif not test_set:
            print('Empty test set')

        elif k > len(training_set):
            print('Expected number of neighbors is higher than number of training data instances')

        else:
            knn(training_set, test_set, k)

    except ValueError as v:
        print(v)

    except FileNotFoundError:
        print('File not found')


  
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
