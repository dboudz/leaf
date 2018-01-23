from flask import Flask, request
from flask_cors import CORS
import base64
import uuid
import os
import json
import tensorflow as tf
import numpy as np

app = Flask(__name__)
# Very important for tests purposes (CORS)
CORS(app)
PATH_STORAGE='/home/input/'



#@app.route("/")
#def hello():
#    return "Hello World!"


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


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


def predict(file_name):
    model_file = "/home/leaf/tensorflow-for-poets-2/tf_files/retrained_graph.pb"
    label_file = "/home/leaf/tensorflow-for-poets-2/tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    NUMBER_OF_TOP_RESULTS = 5

    print "Loading graph results:"
    graph = load_graph(model_file)
    print "Reading tf graph results:"
    tensorImage = read_tensor_from_image_file(file_name,
                                              input_height=input_height,
                                              input_width=input_width,
                                              input_mean=input_mean,
                                              input_std=input_std)

    # print tensorImage
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    # print "Launching model :"
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: tensorImage})
    results = np.squeeze(results)

    # Creating labels
    class_descriptions = []
    labels = load_labels(label_file)
    for s in labels:
        class_descriptions.append(s)
    class_tensor = tf.constant(class_descriptions)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
    classes = table.lookup(tf.to_int64(labels))


    top_k = results.argsort()[-NUMBER_OF_TOP_RESULTS:][::-1]
    values, indices = tf.nn.top_k(results, NUMBER_OF_TOP_RESULTS)

    print "top k :"
    print top_k

    print "Displaying results:"

    prediction_raw = {}
    prediction_sorted = {}
    print values
    print indices

    for i in top_k:
        print labels[i]
        prediction_raw[str(i)] = {"plant": labels[i], "score": format(results[i], '.2f')}
        print results[i]
    # Sorting results :
    for i in range(0,NUMBER_OF_TOP_RESULTS):
        prediction_sorted[str(i+1)] = prediction_raw.get(str(top_k[i]))

    print "prediction raw :"
    print prediction_raw
    return prediction_sorted


# This Endpoint is used for testing purposed
@app.route('/test', methods=['POST'])
def endpoint_predict_test():
    print "Receive request"
    if request.method == 'POST':
        print "Receiving file"
        file = request.files['file']
        print request

        extension = os.path.splitext(file.filename)[1]
        f_uuid= str(uuid.uuid4())
        f_name = f_uuid + extension
        print "Writting file " + PATH_STORAGE + f_name
        file.save(os.path.join(PATH_STORAGE, f_name))

        print "Predicting"
        dictPrediction = predict(PATH_STORAGE + f_name)
        response={'prediction':dictPrediction ,'remote_id':f_uuid}
        print "response is now :"
        print response
        return json.dumps(response)

# Endpoint used by application
@app.route('/predict', methods=['POST'])
def endpoint_predict():

    if request.method == 'POST':
        print "Receiving file"
        jsonrequest=request.get_json()

        file_content = jsonrequest.get('file_content')
        print file_content
        file_extension=jsonrequest.get('file_extension')
        print file_extension

        # Saving file
        f_uuid= str(uuid.uuid4())
        f_name = f_uuid+ '.' + file_extension
        fh = open(PATH_STORAGE+f_name, "wb")
        fh.write(base64.decodestring(file_content))
        fh.close()

        print "Predicting"
        dictPrediction = predict(PATH_STORAGE + f_name)
        response={'prediction':dictPrediction ,'remote_id':f_uuid}
        print "response is now :"
        print response
        return json.dumps(response)
    else:
        print "Receive request that is not a post"
        return json.dumps({'error':'Receive request that is not a POST'})
