from flask import Flask, request, render_template, send_from_directory, jsonify
import sqlite3
from PIL import Image
from flask.wrappers import Response
from Preprocessing import convert_to_image_tensor, invert_image
import torch
from Model import SiameseConvNet, distance_metric
from io import BytesIO
import json
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte, io
from skimage import measure, morphology

app = Flask(__name__, static_folder='./frontend/build/static',
            template_folder='./frontend/build')


def load_model():
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load(
        'Models/model_large_epoch_20', map_location=device))
    return model


def connect_to_db():
    conn = sqlite3.connect('user_signatures1.db')
    return conn


def get_file_from_db(customer_id):
    cursor = connect_to_db().cursor()
    select_fname = """SELECT sign1,sign2,sign3 from signatures1 where customer_id = ?"""
    cursor.execute(select_fname, (customer_id,))
    item = cursor.fetchone()
    cursor.connection.commit()
    return item


def extract_signature(source_image):
    """Extract signature from an input image.
    Parameters
    ----------
    source_image : numpy ndarray
        The pinut image.
    Returns
    -------
    numpy ndarray
        An image with the extracted signatures1.
    """
    # read the input image
    npimg = np.frombuffer(source_image, np.uint8)
    img = npimg
    img = cv2.resize(img, (900, 900))
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    #image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    '''
    # plot the connected components (for debugging)
    ax.imshow(image_label_overlay)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    '''

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)
    print("the_biggest_component: " + str(the_biggest_component))
    print("average: " + str(average))

    # experimental-based ratio calculation, modify it for your cases
    # a4_constant is used as a threshold value to remove connected pixels
    # are smaller than a4_constant for A4 size scanned documents
    a4_constant = (((average/84.0)*250.0)+100)*1.5
    print("a4_constant: " + str(a4_constant))

    # remove the connected pixels are smaller than a4_constant
    b = morphology.remove_small_objects(blobs_labels, a4_constant)

    #b = b.astype(np.unit8)
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    plt.imsave('pre_version.png', b)

    # read the pre-version
    img = cv2.imread('pre_version.png', 0)
    # ensure binary
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # save the the result
    cv2.imwrite("output.png", img)
    return img


def main():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS signatures1 (customer_id TEXT PRIMARY KEY,sign1 BLOB, sign2 BLOB, sign3 BLOB)"""
    cursor = connect_to_db().cursor()
    cursor.execute(CREATE_TABLE)
    cursor.connection.commit()
    DELETE_DATA = """DELETE FROM signatures"""
    cursor1 = connect_to_db().cursor()
    cursor1.execute(DELETE_DATA)
    cursor1.connection.commit()
    # For heroku, remove this line. We'll use gunicorn to run the app
    app.run()  # app.run(debug=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file1 = request.files['uploadedImage1']
    file2 = request.files['uploadedImage2']
    file3 = request.files['uploadedImage3']
    customer_id = request.form['customerID']
    print(customer_id)
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        query = """DELETE FROM signatures1 where customer_id=?"""
        cursor.execute(query, (customer_id,))
        cursor = conn.cursor()
        query = """INSERT INTO signatures1 VALUES(?,?,?,?)"""
        cursor.execute(query, (customer_id, file1.read(),
                       file2.read(), file3.read()))
        conn.commit()
        return jsonify({"error": False})
    except Exception as e:
        print(e)
        return jsonify({"error": True})


@app.route('/verify', methods=['POST'])
def verify():
    try:
        # test = request.json
        # print(test)
        # return test['title']
        customer_id = request.form['customerID']
        file1 = request.files['uploadedImage1']
        file2 = request.files['uploadedImage2']
        file3 = request.files['uploadedImage3']
        # customer_id = request.form['customerID']
        print(customer_id)

        conn = connect_to_db()
        cursor = conn.cursor()
        query = """DELETE FROM signatures1 where customer_id=?"""
        cursor.execute(query, (customer_id,))
        cursor = conn.cursor()
        query = """INSERT INTO signatures1 VALUES(?,?,?,?)"""
        cursor.execute(query, (customer_id, file1.read(),
                       file2.read(), file3.read()))
        conn.commit()

        input_image = Image.open(request.files['newSignature'])
        # img1 = extract_signature(input_image)
        # image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        # lower = np.array([90, 38, 0])
        # upper = np.array([145, 255, 255])
        # mask = cv2.inRange(image, lower, upper)
        # print(input_image)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # close = cv2.morphologyEx(
        #     opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # cnts = cv2.findContours(close, cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # boxes = []
        # for c in cnts:
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     boxes.append([x, y, x+w, y+h])

        # boxes = np.asarray(boxes)
        # left = np.min(boxes[:, 0])
        # top = np.min(boxes[:, 1])
        # right = np.max(boxes[:, 2])
        # bottom = np.max(boxes[:, 3])

        # result[close == 0] = (255, 255, 255)
        # ROI = result[top:bottom, left:right].copy()
        # cv2.rectangle(result, (left, top), (right, bottom), (36, 255, 12), 2)

        # cv2.imshow('result', result)
        # cv2.imshow('ROI', ROI)
        # cv2.imshow('close', close)
        # cv2.imwrite('result.png', result)
        # cv2.imwrite('ROI.png', ROI)
        print(type(input_image))
        input_image_tensor = convert_to_image_tensor(
            invert_image(input_image)).view(1, 1, 220, 155)
        customer_sample_images = get_file_from_db(customer_id)
        # print(customer_sample_images)
        if not customer_sample_images:
            return jsonify({'error': True})
        anchor_images = [Image.open(BytesIO(x))
                         for x in customer_sample_images]
        anchor_image_tensors = [convert_to_image_tensor(invert_image(x)).view(-1, 1, 220, 155)
                                for x in anchor_images]
        model = load_model()

        mindist = math.inf
        for anci in anchor_image_tensors:
            f_A, f_X = model.forward(anci, input_image_tensor)
            dist = float(distance_metric(f_A, f_X).detach().numpy())
            mindist = min(mindist, dist)

            if dist <= 0.145139:  # Threshold obtained using Test.py
                return jsonify({"match": True, "error": False, "threshold": "%.6f" % (0.145139), "distance": "%.6f" % (mindist)})
        return jsonify({"match": False, "error": False, "threshold": 0.145139, "distance": round(mindist, 6)})
    except Exception as e:
        print(e)
        return jsonify({"error": True})


@app.route("/manifest.json")
def manifest():
    return send_from_directory('./frontend/build', 'manifest.json')


@app.route("/favicon.ico")
def favicon():
    return send_from_directory('./frontend/build', 'favicon.ico')


if __name__ == '__main__':
    main()
