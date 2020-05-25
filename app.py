import os
from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app =Flask(__name__)


@app.route("/",methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route("/index/")
def index1():
    return render_template('index.html')

@app.route("/research/")
def research():
    return render_template('research.html')

@app.route("/cancertypes/")
def cancertypes():
    return render_template('cancertypes.html')

@app.route("/hl/")
def hl():
    return render_template('hl.html')

@app.route("/support/")
def support():
    return render_template('support.html')

def predict():
    target = os.path.join(APP_ROOT, 'temp1/')
    preprocessed_data_folder = target
    batch_size = 10 # divide set into batches to prevent machine from running out of memory
    data_shape = [250, 350, 400]

    patients = os.listdir(preprocessed_data_folder)
    patients.sort()

    test_batches = []
    current_test_batch = []
    test_data_size = 0
    for i in range(len(patients)):  
        if patients[i].startswith('.'): continue # ignore hidden files
            
        if np.load(preprocessed_data_folder + patients[i])['set'] == 'test':
            current_test_batch.append(patients[i])
            if (len(current_test_batch) == batch_size): 
                test_batches.append(current_test_batch)
                test_data_size += batch_size
                current_test_batch = []
    if len(current_test_batch) != 0:
        test_data_size += len(current_test_batch)
        # pad zeros to make its size equal to batch_size
        while (len(current_test_batch) != batch_size):
            current_test_batch.append(0)
        test_batches.append(current_test_batch)
        current_test_batch = []
    test_batches = np.array(test_batches)

   


    def read_batch(batch_files):

        # do not account the placeholder zeros for the batch size
        current_batch_size = batch_size - np.sum(batch_files == '0')

        batch_features = np.zeros((current_batch_size, data_shape[0], data_shape[1], data_shape[2], 1))
        batch_ids = []

        for i in range(len(batch_files)):
            if batch_files[i] != '0':
                data = np.load(preprocessed_data_folder + batch_files[i])
                batch_features[i,:,:,:,0] = data['data']
                batch_ids.append(batch_files[i][:32])

        return batch_features, batch_ids

    test_features_batch_sample, test_ids_batch_sample = read_batch(test_batches[0])
    
    saved = os.path.join(APP_ROOT, 'static/temp2')
    
    plt.imshow(test_features_batch_sample[0,:,:,100,0], cmap=plt.cm.bone)
    my_file = 'graph.png'
    
    plt.savefig(os.path.join(saved, my_file)) 
    
    
    
    
    
    
   
    target1 = os.path.join(APP_ROOT, 'models/completemodel_3')
    model_save_path = target1
    prediction_save_path = 'C:/Users/sahana/Desktop/data/result.csv'
    
    cnn_graph = tf.Graph()
    prediction_file = open(prediction_save_path, 'w')
    prediction_file.write('id,cancer\n') # header line
    with tf.Session(graph=cnn_graph) as sess:
        # load model
        loader = tf.train.import_meta_graph(model_save_path + '.meta')
        loader.restore(sess, model_save_path)
    
        # obtain tensors
        x = cnn_graph.get_tensor_by_name('x:0')
        y = cnn_graph.get_tensor_by_name('y:0')
        keep_prob = cnn_graph.get_tensor_by_name('keep_prob:0')
        logits = cnn_graph.get_tensor_by_name('logits:0')
        for batch_index in range(test_batches.shape[0]):
            test_features_batch, test_ids_batch = read_batch(test_batches[batch_index])
            predictions = sess.run(tf.nn.softmax(logits), feed_dict={
                x: test_features_batch,
                keep_prob: 1.
            })
   
            for test_index in range(len(test_ids_batch)):
                # save predictions to file
                prediction_file.write(test_ids_batch[test_index] + ',' + str(predictions[test_index,1]) + '\n')
                # print out predictions
            
                x=test_ids_batch[test_index]
                y=float(predictions[test_index,1])
    return y*100
@app.route("/predict",methods=["GET","POST"])
def upload():
    target = os.path.join(APP_ROOT, 'temp1/')
    if request.method == 'POST':
        file = request.files['image'] # 'image' is the id passed in input file form field
        filename = file.filename
        # filename = filename(filename)
        file.save("".join([target, filename])) #saving file in temp folder
        print("upload Completed")
        pred1=predict()
        pred = "{:.2f}".format(pred1)
        return render_template('result1.html',result=pred)



if __name__ == '__main__':
    app.run()