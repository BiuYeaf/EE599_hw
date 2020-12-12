import numpy as np
import matplotlib.pyplot as plt 


def hidden_layer(input,output):
    w = np.random.normal(0, 2 / input, (output, input))
    b = np.zeros((output, 1))
    return w,b

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t
def tanh_div(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return dt
def ReLu(x):
    return np.maximum(0,x)

def ReLU_div(x):
    return (x > 0).astype(float)

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)[:, None].T
def CrossEntropy(y, a):
    return -np.sum(y * np.log(a))
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = Sigmoid(x)
    ds = s*(1-s)
    return ds

def update_Sigmoid(x,y,batch_size,error,l,w1,b1,w2,b2,w3,b3):
    regular = 1e-5
    x1=Sigmoid(w1@x+b1)
    x2=Sigmoid(w2@x1+b2)
    x3=Softmax(w3@x2+b3)
    
    w3_update=w3-(l*error@(x2.T))/batch_size - 2* regular * w3
    b3_update=b3-l*np.sum(error,axis=1)[:,None]/batch_size
    
    error2=sigmoid_derivative(w2@x1+b2)*(w3.T@error)
    w2_update=w2-(l*error2@(x1.T))/batch_size- 2* regular * w2
    b2_update=b2-l*np.sum(error2,axis=1)[:,None]/batch_size
    
    error3=sigmoid_derivative(w1@x+b1)*(w2.T@error2)
    w1_update=w1-(l*error3@(x.T))/batch_size- 2* regular * w1
    b1_update=b1-l*np.sum(error3,axis=1)[:,None]/batch_size
    return w3_update,b3_update,w2_update,b2_update,w1_update,b1_update

def feedprop_Sigmoid(x,y,w1,b1,w2,b2,w3,b3):
    x=np.reshape(x,(784,1))
    y=np.reshape(y,(10,1))
    x1=Sigmoid(w1@x+b1)
    x2=Sigmoid(w2@x1+b2)
    x3=Softmax(w3@x2+b3)
    error=x3-y
    return error

def eval_model_Sigmoid(xdata,ydata,w1,b1,w2,b2,w3,b3):
    x1=Sigmoid(w1@xdata.T+b1)
    x2=Sigmoid(w2@x1+b2)
    x3=Softmax(w3@x2+b3)
    x3=x3.T
    
    accuracy=0.0
    for i in range(xdata.shape[0]):
        if(np.argmax(x3[i])==np.argmax(ydata[i])):
            accuracy+=1
    accuracy=accuracy/ydata.shape[0]
    loss=CrossEntropy(ydata,x3)/ydata.shape[0]
    return accuracy,loss

def nn(x_train,x_val,y_train,y_val,learning_rate,batch_size,epoch=1):
    np.random.seed(seed=24)
    w1,b1=hidden_layer(784,128)
    w2,b2=hidden_layer(128,64)
    w3,b3=hidden_layer(64,10)
    
    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    
    for times in range(epoch):
        for updates in range(int(x_train.shape[0]/batch_size)):
            total_error=np.zeros((500,10,1))
            for data in range(batch_size):
                x=x_train[data+batch_size*updates]
                y=y_train[data+batch_size*updates]
                total_error[data]=feedprop_Sigmoid(x,y,w1,b1,w2,b2,w3,b3) 
            w3,b3,w2,b2,w1,b1=update_Sigmoid(x_train[updates*batch_size:(updates+1)*batch_size,:].T,
                                          y_train[updates*batch_size:(updates+1)*batch_size,:].T,
                                          batch_size,np.reshape(total_error,(500,10)).T,learning_rate,
                                          w1,b1,w2,b2,w3,b3)
                   
        train_accuracy,train_loss=eval_model_Sigmoid(x_train,y_train,w1,b1,w2,b2,w3,b3)
        val_accuracy,val_loss=eval_model_Sigmoid(x_val,y_val,w1,b1,w2,b2,w3,b3)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)
        
        
        if(times==19 or times==39):
            learning_rate=learning_rate/2
        print('Epoch:%d,Train Loss:%f,Training Accuracy:%f,Validation Loss:%f,Validation Accuracy:%f'%(times+1,train_loss,train_accuracy,val_loss,val_accuracy))
    return train_loss_list,train_acc_list,val_loss_list,val_acc_list,w1,w2,w3,b1,b2,b3

learning_rate=0.01
batch_size=500
train_loss,train_acc,val_loss,val_acc,w1,w2,w3,b1,b2,b3=nn(x_train,x_val,y_train,y_val,learning_rate,batch_size,epoch=50)


plt.figure()
plt.subplot(211)
plt.title('Learning rate=0.01,ReLu')
plt.xlabel('epoch')
plt.ylabel('accuracy rate')
plt.plot(train_acc,c='r',label='training accuracy')
plt.plot(val_acc,c='b',label='validation accuracy')
plt.legend()
plt.subplot(212)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.plot(train_loss,c='r',label='training loss')
plt.plot(val_loss,c='b',label='validation loss')
plt.legend()
plt.show()