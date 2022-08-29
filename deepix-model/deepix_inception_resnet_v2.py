import urllib3
import os
import pandas as pd
import pymysql
import requests
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2
import redis
import gc
import pprint
import time
import datetime
AUTOTUNE = tf.data.experimental.AUTOTUNE



def label_str_to_tensor(labelStr):
    return tf.convert_to_tensor(list(map(float, bytes.decode(labelStr.numpy()).split(','))), dtype=tf.float32)


def load_and_preprocess_image_from_url(url):
    try:
        image = tf.io.decode_image(httpclient.request('GET',bytes.decode(url.numpy())).data, channels=3)
        image = tf.image.resize(image, [299, 299])
        image /= 255.0
        return image
    except Exception as e:
        print(url)
    


def load_and_preprocess_image_from_url_warp(url):
    return tf.py_function(load_and_preprocess_image_from_url, [url], Tout=(tf.float32))


def build_label_data(*args):
    return [label_str_to_tensor(i) for i in args]


def build_label_data_warp(bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label, label):
    return tf.py_function(build_label_data,
                          [bookmark_label, view_label, sanity_label, restrict_label, x_restrict_label, label],
                          Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

def build_data_set(deepix_train_index,offset,batch_size):
    time_start=time.time()
    data_from_db = pd.read_sql(sql, db_connection,params=[deepix_train_index,deepix_train_index+offset])
    time_end=time.time()
    print('查询sql耗时：',time_end-time_start,'s')
    img_dataset = tf.data.Dataset.from_tensor_slices(data_from_db.img_path.values)
    label_dataset = tf.data.Dataset.from_tensor_slices((data_from_db.bookmark_label.values, data_from_db.view_label.values,data_from_db.sanity_label.values, data_from_db.restrict_label.values,data_from_db.x_restrict_label.values, data_from_db.label.values))
    img_dataset = img_dataset.map(load_and_preprocess_image_from_url_warp, num_parallel_calls=AUTOTUNE)
    label_dataset = label_dataset.map(build_label_data_warp, num_parallel_calls=AUTOTUNE)
    dataset = tf.data.Dataset.zip((img_dataset, label_dataset))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.map(_fixup_shape, num_parallel_calls=AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    time_end=time.time()
    del data_from_db
    gc.collect()
    print('构建dataset耗时：',time_end-time_start,'s')
    return dataset
    
    
def build_model(lr):
    multi_loss = {'bookmark_predict': 'categorical_crossentropy', 'view_predict': 'categorical_crossentropy',
              'sanity_predict': 'categorical_crossentropy', 'restrict_predict': 'categorical_crossentropy',
              'x_restrict_predict': 'categorical_crossentropy', 'tag_predict': 'binary_crossentropy'}
    multi_metrics = {'bookmark_predict': 'acc', 'view_predict': 'acc', 'sanity_predict': 'acc', 'restrict_predict': 'acc',
                 'x_restrict_predict': 'acc', 'tag_predict': 'acc'}
    model = inception_resnet_v2()
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            #optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
            loss=multi_loss,
              # 权重需要在调整
              loss_weights=[1, 1,1,1,1,1000],
              # metrics=['acc']
              metrics=multi_metrics,
              )
    return model
    
def build_cp_callback(checkpoint_path):  
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_weights_only=True,save_freq=5 * batch_size)
    return cp_callback;

def _fixup_shape(images, labels):
    images.set_shape([None, None, None, 3])
    labels[0].set_shape([None, 10])
    labels[1].set_shape([None, 10])
    labels[2].set_shape([None, 10])
    labels[3].set_shape([None, 3])
    labels[4].set_shape([None, 3])
    labels[5].set_shape([None, 10240])
    return images, labels

httpclient=urllib3.PoolManager()
lr = 0.001

batch_size = 32
epoch = 1
checkpoint_path = "model-checkpoint/cp-{epoch:04d}.ckpt" 
save_weight_path='./model_weight/deepix_model_weight'
save_weight_history_path='./model_weight_history/'
redis_index_key='deepix_inception_resnet_v2_train_index_epoch'
max_deepix_train_index=90000000
#数据集构建
#获取索引
redis_conn = redis.Redis(host='127.0.0.1', port= 6379, password= '', db= 0)
deepix_train_index =  int(redis_conn.get(redis_index_key))
pp = pprint.PrettyPrinter(indent=4)
if(deepix_train_index is None):
    deepix_train_index=0
print('当前训练到'+str(deepix_train_index))
sql = '''
select img_path,

           REGEXP_REPLACE(bookmark_label, '\\\\[|\\\\]', '') as bookmark_label ,
              REGEXP_REPLACE(view_label, '\\\\[|\\\\]', '') as view_label,
              REGEXP_REPLACE(sanity_label, '\\\\[|\\\\]', '') as sanity_label,
              REGEXP_REPLACE(restrict_label, '\\\\[|\\\\]', '') as restrict_label,
              REGEXP_REPLACE(x_restrict_label, '\\\\[|\\\\]', '') as x_restrict_label,
              REGEXP_REPLACE(label, '\\\\[|\\\\]', '') as label

from deepix_data 
where illust_id between %s and %s  order by illust_id 
'''
db_connection = pymysql.connect(host="10.0.0.5", user="root", password="Cheerfun.dev", db="deepix")



offset=800000
model=build_model(lr)
model.load_weights(save_weight_path)
#cp_callback=build_cp_callback(checkpoint_path);
epoch_index=int(redis_conn.get('deepix_inception_resnet_v2_train_epoch_index'))
while(True
        #deepix_train_index<=max_deepix_train_index
        #and datetime.datetime.now().hour>=6 and datetime.datetime.now().hour<=19
        ):
    if(deepix_train_index>40000000):
        offset=400000
    else:
        offset=800000
    print(time.asctime()+' 开始训练索引'+str(deepix_train_index))
    if(deepix_train_index>=max_deepix_train_index ):
        print('第'+str(epoch_index)+'轮训练结束')
        model.save(save_weight_history_path+'epoch-'+str(epoch_index)+'.h5')
        print('第'+str(epoch_index)+'个模型历史参数保存完毕')
        epoch_index+=1
        redis_conn.set('deepix_inception_resnet_v2_train_epoch_index',epoch_index)
        deepix_train_index=5000000
        redis_conn.set(redis_index_key,deepix_train_index)
        continue
    dataset=build_data_set(deepix_train_index,offset,batch_size)
    
    history = model.fit(dataset, epochs=epoch, )
    model.save_weights(save_weight_path)
    deepix_train_index+=offset
    redis_conn.set(redis_index_key,deepix_train_index)
    del dataset
    #gc.collect()
    

