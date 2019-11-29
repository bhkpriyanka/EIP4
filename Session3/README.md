**Final Validation accuracy for Base Network (After 50 epochs)**: 82.64%

**ModelDefinition:**   
model = Sequential()  
model.add(SeparableConv2D(filters=32,kernel_size=3,activation='relu',use_bias =False,padding='same',input_shape=(32, 32, 3))) #RF=3;Cout=32x32x32  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=64,kernel_size=3,activation='relu',use_bias =False,padding='same')) #RF=5;Cout=32x32x64  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=100,kernel_size=3,activation='relu',use_bias =False,padding='same')) #RF=7;Cout=32x32x100  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(MaxPooling2D(pool_size=(2,2))) #RF=8;Cout=16x16x100   
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=64,kernel_size=3,activation='relu',use_bias =False,padding='same')) #RF=12;Cout=16x16x64  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=100,kernel_size=3,activation='relu',use_bias =False,padding='same')) #RF=16;Cout=16x16x100  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=150,kernel_size=3,activation='relu',use_bias =False,padding='same')) #RF=20;Cout=16x16x150  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(MaxPooling2D(pool_size=(2,2))) #RF=22;Cout=8x8x150  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=100,kernel_size=3,activation='relu',use_bias =False,padding='same')) #RF=30;Cout=8x8x100  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
model.add(SeparableConv2D(filters=200,kernel_size=3,activation='relu',use_bias =False,padding='valid')) #RF=38;Cout=6X6X200  
model.add(BatchNormalization())  
model.add(Dropout(0.1))  
  
model.add(GlobalAveragePooling2D()) #200  
model.add(Dense(64, activation='relu',use_bias =False)) #64  
model.add(Dense(num_classes, activation='softmax',use_bias =False)) #10  
  
**50 epoch logs** **Beats base validation in 27th epoch and highest Vacc=84.83**  
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.006.
390/390 [==============================] - 27s 70ms/step - loss: 1.5019 - acc: 0.4510 - val_loss: 2.8853 - val_acc: 0.3993
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0045489007.
390/390 [==============================] - 23s 60ms/step - loss: 1.1219 - acc: 0.6010 - val_loss: 1.2414 - val_acc: 0.6004
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0036630037.
390/390 [==============================] - 23s 60ms/step - loss: 0.9614 - acc: 0.6604 - val_loss: 0.9373 - val_acc: 0.6871
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0030659172.
390/390 [==============================] - 23s 60ms/step - loss: 0.8635 - acc: 0.6947 - val_loss: 0.7840 - val_acc: 0.7301
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0026362039.
390/390 [==============================] - 23s 60ms/step - loss: 0.7915 - acc: 0.7221 - val_loss: 0.7403 - val_acc: 0.7504
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0023121387.
390/390 [==============================] - 23s 59ms/step - loss: 0.7414 - acc: 0.7406 - val_loss: 0.7311 - val_acc: 0.7545
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0020590254.
390/390 [==============================] - 23s 60ms/step - loss: 0.7022 - acc: 0.7533 - val_loss: 0.6424 - val_acc: 0.7808
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0018558614.
390/390 [==============================] - 23s 59ms/step - loss: 0.6686 - acc: 0.7658 - val_loss: 0.6544 - val_acc: 0.7720
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0016891892.
390/390 [==============================] - 23s 59ms/step - loss: 0.6434 - acc: 0.7744 - val_loss: 0.6607 - val_acc: 0.7803
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0015499871.
390/390 [==============================] - 23s 59ms/step - loss: 0.6226 - acc: 0.7814 - val_loss: 0.5900 - val_acc: 0.7976
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0014319809.
390/390 [==============================] - 23s 60ms/step - loss: 0.6027 - acc: 0.7889 - val_loss: 0.5820 - val_acc: 0.7982
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.001330672.
390/390 [==============================] - 23s 60ms/step - loss: 0.5822 - acc: 0.7952 - val_loss: 0.5695 - val_acc: 0.8055
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0012427506.
390/390 [==============================] - 23s 59ms/step - loss: 0.5678 - acc: 0.8011 - val_loss: 0.5958 - val_acc: 0.8024
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0011657276.
390/390 [==============================] - 23s 59ms/step - loss: 0.5571 - acc: 0.8050 - val_loss: 0.5560 - val_acc: 0.8102
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0010976948.
390/390 [==============================] - 23s 60ms/step - loss: 0.5401 - acc: 0.8116 - val_loss: 0.5423 - val_acc: 0.8121
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0010371651.
390/390 [==============================] - 23s 59ms/step - loss: 0.5327 - acc: 0.8137 - val_loss: 0.5632 - val_acc: 0.8057
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000982962.
390/390 [==============================] - 23s 59ms/step - loss: 0.5267 - acc: 0.8147 - val_loss: 0.5776 - val_acc: 0.8056
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0009341429.
390/390 [==============================] - 23s 59ms/step - loss: 0.5110 - acc: 0.8207 - val_loss: 0.5325 - val_acc: 0.8161
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0008899436.
390/390 [==============================] - 23s 58ms/step - loss: 0.5116 - acc: 0.8211 - val_loss: 0.5968 - val_acc: 0.7995
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000849738.
390/390 [==============================] - 23s 58ms/step - loss: 0.4986 - acc: 0.8254 - val_loss: 0.5426 - val_acc: 0.8136
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0008130081.
390/390 [==============================] - 23s 58ms/step - loss: 0.4932 - acc: 0.8264 - val_loss: 0.5205 - val_acc: 0.8210
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000779322.
390/390 [==============================] - 23s 58ms/step - loss: 0.4818 - acc: 0.8292 - val_loss: 0.5362 - val_acc: 0.8174
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0007483163.
390/390 [==============================] - 23s 58ms/step - loss: 0.4730 - acc: 0.8333 - val_loss: 0.5280 - val_acc: 0.8223
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0007196833.
390/390 [==============================] - 23s 59ms/step - loss: 0.4738 - acc: 0.8340 - val_loss: 0.5601 - val_acc: 0.8135
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0006931608.
390/390 [==============================] - 23s 59ms/step - loss: 0.4672 - acc: 0.8353 - val_loss: 0.5253 - val_acc: 0.8256
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0006685237.
390/390 [==============================] - 23s 58ms/step - loss: 0.4566 - acc: 0.8408 - val_loss: 0.5254 - val_acc: 0.8250
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0006455778.
390/390 [==============================] - 23s 58ms/step - loss: 0.4521 - acc: 0.8414 - val_loss: 0.5065 - val_acc: 0.8307
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0006241548.
390/390 [==============================] - 23s 59ms/step - loss: 0.4514 - acc: 0.8409 - val_loss: 0.5067 - val_acc: 0.8320
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0006041079.
390/390 [==============================] - 23s 59ms/step - loss: 0.4490 - acc: 0.8412 - val_loss: 0.4927 - val_acc: 0.8334
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0005853088.
390/390 [==============================] - 23s 58ms/step - loss: 0.4405 - acc: 0.8429 - val_loss: 0.4877 - val_acc: 0.8363
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0005676443.
390/390 [==============================] - 23s 59ms/step - loss: 0.4397 - acc: 0.8445 - val_loss: 0.4888 - val_acc: 0.8359
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0005510148.
390/390 [==============================] - 23s 58ms/step - loss: 0.4332 - acc: 0.8462 - val_loss: 0.4796 - val_acc: 0.8355
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0005353319.
390/390 [==============================] - 23s 59ms/step - loss: 0.4319 - acc: 0.8485 - val_loss: 0.4767 - val_acc: 0.8379
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.000520517.
390/390 [==============================] - 23s 59ms/step - loss: 0.4242 - acc: 0.8502 - val_loss: 0.5191 - val_acc: 0.8254
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0005065001.
390/390 [==============================] - 23s 58ms/step - loss: 0.4229 - acc: 0.8510 - val_loss: 0.5159 - val_acc: 0.8296
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0004932182.
390/390 [==============================] - 23s 58ms/step - loss: 0.4205 - acc: 0.8512 - val_loss: 0.5074 - val_acc: 0.8333
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0004806152.
390/390 [==============================] - 23s 58ms/step - loss: 0.4195 - acc: 0.8512 - val_loss: 0.4841 - val_acc: 0.8387
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0004686402.
390/390 [==============================] - 23s 59ms/step - loss: 0.4157 - acc: 0.8523 - val_loss: 0.4859 - val_acc: 0.8384
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0004572474.
390/390 [==============================] - 22s 58ms/step - loss: 0.4112 - acc: 0.8560 - val_loss: 0.5077 - val_acc: 0.8327
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0004463954.
390/390 [==============================] - 22s 57ms/step - loss: 0.4080 - acc: 0.8550 - val_loss: 0.5019 - val_acc: 0.8351
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0004360465.
390/390 [==============================] - 22s 57ms/step - loss: 0.4095 - acc: 0.8559 - val_loss: 0.4874 - val_acc: 0.8410
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0004261666.
390/390 [==============================] - 22s 57ms/step - loss: 0.4022 - acc: 0.8586 - val_loss: 0.4551 - val_acc: 0.8483
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0004167245.
390/390 [==============================] - 22s 56ms/step - loss: 0.4070 - acc: 0.8549 - val_loss: 0.4695 - val_acc: 0.8458
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0004076918.
390/390 [==============================] - 22s 56ms/step - loss: 0.3972 - acc: 0.8588 - val_loss: 0.5035 - val_acc: 0.8362
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0003990423.
390/390 [==============================] - 22s 56ms/step - loss: 0.3994 - acc: 0.8597 - val_loss: 0.4835 - val_acc: 0.8415
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0003907522.
390/390 [==============================] - 22s 56ms/step - loss: 0.3976 - acc: 0.8601 - val_loss: 0.4673 - val_acc: 0.8436
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0003827995.
390/390 [==============================] - 22s 56ms/step - loss: 0.3917 - acc: 0.8627 - val_loss: 0.4841 - val_acc: 0.8401
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0003751641.
390/390 [==============================] - 22s 56ms/step - loss: 0.3931 - acc: 0.8615 - val_loss: 0.4860 - val_acc: 0.8401
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0003678274.
390/390 [==============================] - 22s 56ms/step - loss: 0.3866 - acc: 0.8633 - val_loss: 0.4769 - val_acc: 0.8437
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0003607721.
390/390 [==============================] - 22s 56ms/step - loss: 0.3842 - acc: 0.8628 - val_loss: 0.4633 - val_acc: 0.8461
Model took 1141.98 seconds to train

Accuracy on test data is: 84.61
