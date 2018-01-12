from flask import Flask, request
from flask_cors import CORS
import uuid
import os
import json
import tensorflow as tf
import numpy as np
import base64
app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return "Hello World!"



def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


@app.route('/predict', methods=['POST'])
def upload():

    if  request.method == 'POST':
        print "Receive POST request"
        #print request.form
        print 'GEt json'
        print request.get_json()
        print "Receiving file"
        jsonrequest=request.get_json()

        file_content = jsonrequest.get('file_content')
        print file_content
        file_extension=jsonrequest.get('file_extension')
        print file_extension

        f_name = str(uuid.uuid4()) +"."+ file_extension
        fh = open("/home/leaf/flask/"+f_name, "wb")
        #fh.write(file_content.decode('base64'))
        #fh.write(file_content.decodebytes('base64'))

        #fh.write(base64.b64decode(file_content).decode('UTF-8'))

        fh.write(base64.decodestring(file_content))
        fh.write("lol")
        fh.close()

        return json.dumps({'lol':'lol'})
    else:
        print "Receive request that is not a post"
        return json.dumps({'error':'error'})
