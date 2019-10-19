"""
this file creates a model of the face

"""
####################################################libraries########################################################
import pandas as pd
import time
import numpy
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection  import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint
###################################################################################################################


############################################variables#########################################################
##path to store model
filepath='E:/software/MCA 6TH project/stranger detection system/model/'
##path to load database
df=pd.read_csv('E:/software/MCA 6TH project/stranger detection system/database/preprocessed/full preprocessed/all.csv',header= None)
X=df.iloc[:,0:22].values##Xvalues
Y=df.iloc[:,22].values##classes
##############################################################################################################

###########################################encode  categorical values#########################################
encoder = LabelBinarizer()
y = encoder.fit_transform(Y)
yclasses=encoder.classes_##classes in it the csv file
save_names=yclasses
#############################################################################################################


################################################################split training data#################################
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.0,random_state=42)
################################################################################################################


##################################################classifier#######################################################
classifier=Sequential()
################################################################################################################


######################################################layers########################################################
sgd=optimizers.SGD(lr=0.005,nesterov=True)##Stochastic Gradient Descent
classifier.add(Dense(output_dim =16,init = 'uniform',activation='relu',input_dim=22))
#classifier.add(Dropout(rate=0.01))
classifier.add(Dense(output_dim =14,init = 'uniform',activation='relu'))
classifier.add(Dense(output_dim =12,init = 'uniform',activation='relu'))
classifier.add(Dense(output_dim =10,init = 'uniform',activation='relu'))

classifier.add(Dense(output_dim =len(save_names),init = 'uniform',activation='sigmoid'))
##################################################################################################################


#########################################################compiling the model######################################
classifier.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
print('done')
################################################################################################################


################################################################checkpoint#########################################
checkpoint=ModelCheckpoint(filepath+'chekpoints.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
###################################################################################################################


##############################################################fit and train model####################################
start_time=time.time()
classifier.fit(X_train,y_train,batch_size=10,callbacks=[checkpoint],nb_epoch=1000)
total_time=time.time()-start_time
elapsed_time=time.strftime("%H:%M:%S", time.gmtime(total_time))
print(elapsed_time)
#######################################################################################################################


#####################################################save model and names##############################################
classifier.save(filepath+'face_recognition.h5',overwrite=True, include_optimizer=True)
with open(filepath+'savednames.csv','w') as f_handle:##save a file
                        numpy.savetxt(f_handle,save_names,fmt='%s',delimiter=',',header='names')
######################################################################################################################

"""
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in y_pred ]
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
test=encoder.inverse_transform(y_test)
pred=encoder.inverse_transform(y_pred)"""