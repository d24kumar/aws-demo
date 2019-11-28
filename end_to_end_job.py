from __future__ import absolute_import, division, print_function, unicode_literals

import boto3
import pandas as pd
import sys
from boto3.dynamodb.conditions import Key, Attr
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

def upload_accuracy(UserId, accuracy):
    accuracy_str = str(accuracy)
    dynamodb = boto3.resource('dynamodb')
    dynamodb_table = dynamodb.Table('ContinyUserIdTable')
    response = dynamodb_table.put_item(
        Item={
            'UserId' : UserId,
            'Accuracy': accuracy_str
        }
    )

def upload_to_s3(filename):
    s3 = boto3.client('s3')
    s3.upload_file(filename, 'continy-ub-model-bucket', filename)


def getModel(UserId):
    model_name = 'model_' + str(UserId) + '.tflite'
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    train_dataset_fp = str(UserId) + '.csv'
    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    # column order in CSV file
    column_names = [ 'Day', 'Hour','DeviceId','ProviderId', 'action']
    feature_names = column_names[:-1]
    label_name = column_names[-1]
    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))
    class_names = ['play', 'not_play']

    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)

    features, labels = next(iter(train_dataset))

    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels


    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(2)
    ])


    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    def loss(model, x, y):
        y_ = model(x)
        return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


#to minimize the loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                              loss(model, features, labels).numpy()))


    ## Note: Rerunning this cell uses the same model variables

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    epoch_accuracy_arr = []
    num_epochs = 101

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(y, model(x))

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 10 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
            epoch_accuracy_arr.append(epoch_accuracy.result())
    #convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(model_name, "wb").write(tflite_model)

    max_accuracy = float(max(epoch_accuracy_arr))
    max_accuracy = max_accuracy*100
    max_accuracy = round(max_accuracy, 2)
    print('%.2f'%max_accuracy)
    upload_to_s3(model_name)
    upload_accuracy(UserId, max_accuracy)


def get_user_data(user_id):
    dynamodb_res = boto3.resource('dynamodb')
    dynamodb_table = dynamodb_res.Table('ContinyActivityDB')


    response = dynamodb_table.query(
        KeyConditionExpression=Key('UserId').eq(user_id)
    )
    print('hello')
    a = []
    tags = [len(response['Items']), 4, "play", "not_play"]
    a.append(tags)
    for item in response['Items']:
        b = []
        b.append(item['Day'])
        b.append(item['Hour'])
        b.append(item['DeviceId'])
        b.append(item['ProviderId'])
        b.append(item['action'])

        a.append(b)
        print(item)


    var = response['Items']
    print(len(var))
    print(a)
    pd_dataframe = pd.DataFrame(a)
    print(pd_dataframe.head())
    csv_name = user_id + '.csv'
    pd_dataframe.to_csv(csv_name, header=False, index=False)



def main():
    strA = sys.argv[1]
    strA = strA.split(',')
    for item in strA:
        print(item)
        get_user_data(item)
        getModel(item)


main()
