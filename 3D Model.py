from utlis import *
from sklearn.model_selection import train_test_split

path='myData'
data=importDatainfo(path)

data = balanceData(data,display=True)
imagesPath,steerings=loadData(path,data)
print(imagesPath[0],steerings[0])


xtrain,xval,ytrain,yval=train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)

print (xtrain.shape[0])


model = createModel()
model.summary()

history=model.fit(batchGen(xtrain,ytrain,100,1),steps_per_epoch=500,epochs=10,validation_data=batchGen(xval,yval,100,0),validation_steps=200)

model.save('final2.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.show()



