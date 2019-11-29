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
