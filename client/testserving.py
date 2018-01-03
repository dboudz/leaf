#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:21:52 2017

@author: dbo
"""
import cPickle as pickle
import urllib
from google.protobuf.json_format import MessageToJson




from tensorflow.python.saved_model import signature_constants
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)

  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  return result


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
#tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS



def predict(image_path="/Users/dboudeau/Downloads/1.jpg"):
    host = "localhost"
    port = 9000
    model_name = "serving_default"

    channel = implementations.insecure_channel(host, int(port))
    # J'ai l"impression que le stub est le meme pour predic & classif
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    # Je ne sais pas pourquoi ca ne marche pas ...
    #request.model_spec.signature_name=signature_constants.PREDICT_METHOD_NAME

    proto=tf.contrib.util.make_tensor_proto(read_tensor_from_image_file(image_path))

    request.inputs["images"].CopyFrom(proto)
    #request.inputs["inputs"].CopyFrom(proto)

#    for k,v in model_input.items():
#        request.inputs[k].CopyFrom(
#            tf.contrib.util.make_tensor_proto(v))

    result = stub.Predict(request, 10.0)  # 10 secs timeout
#    resul=stub.Classify(request,10.0)
    jsonObj = MessageToJson(result)
    print result
    return jsonObj

predict()



