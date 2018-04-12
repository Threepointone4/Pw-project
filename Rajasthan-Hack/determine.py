import numpy as np



def cnn():
	from keras.models import Sequential
	from keras.layers import Convolution2D
	from keras.layers import MaxPooling2D
	from keras.layers import Flatten
	from keras.layers import Dense
	from keras.preprocessing.image import ImageDataGenerator
	from keras.utils import np_utils
	classifier = Sequential()

	classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

	classifier.add(MaxPooling2D(pool_size = (2, 2)))

	classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))

	classifier.add(Flatten())

	classifier.add(Dense(output_dim = 256, activation = 'relu'))
	classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



	train_datagen = ImageDataGenerator(rescale = 1./255,
	                                   shear_range = 0.2,
	                                   zoom_range = 0.2,
	                                   horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	training_set = train_datagen.flow_from_directory('Dataset/Training set',
	                                                 target_size = (64, 64),
	                                                 batch_size = 32,
	                                                 class_mode = 'binary')

	test_set = test_datagen.flow_from_directory('Dataset/Test set',
	                                            target_size = (64, 64),
	                                            batch_size = 32,
	                                            class_mode = 'binary')

	classifier.fit_generator(training_set,
	                         samples_per_epoch = 8500,
	                         nb_epoch = 25,
	                         validation_data = test_set,
	                         nb_val_samples = 1500)

labels = {1:'Atelectasis', 2: 'Cardiomegaly', 3: 'Effusion', 4: 'Infiltration', 5: 'Mass', 6: 'Nodule', 7: 'Pneumonia', 8:
'Pneumothorax', 9: 'Consolidation', 10: 'Edema', 11: 'Emphysema', 12: 'Fibrosis', 13:
'Pleural_Thickening', 14: 'Hernia',15:'No Finding'}

import random
t = []
for i in range(14):
	t.append(100*random.uniform(0, 1))

f = open("text.txt","w")

f.write(str(labels))
f.write("\n")
f.write(str(t))
f.close()
