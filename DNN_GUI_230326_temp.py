import streamlit as st
import pandas as pd
import numpy as np
#import os
#import csv
import tensorflow as tf
#import math
import altair as alt
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
#from PIL import Image
from sklearn.metrics import mean_squared_error

#============================================================================
class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))
get_custom_objects().update({'Mish': Mish(mish)})
#============================================================================
def DNN_func():
        my_bar = st.progress(0)
        #sequential
        model=tf.keras.models.Sequential()
        model.add(layers.Dense(input_shape=(in_num,),name='layer1',units=hid1_unit_num,activation=acti1))
        model.add(layers.Dense(name='layer2',units=hid2_unit_num,activation=acti2))
        model.add(layers.Dense(name='layer3',units=hid3_unit_num,activation=acti3))
        model.add(layers.Dense(name='layer4',units=hid4_unit_num,activation=acti4))
        model.add(layers.Dense(name='layer5',units=hid5_unit_num,activation=acti5))
        ######################################################################################
        model.add(layers.Dense(name='layout',units=out_num,activation=acti6))
        #model
        model.summary()
        LOSS='mse'
        METRICS=['mae']
        OPTIMIZER=tf.keras.optimizers.SGD
        LEARNING_RATE=float(0.01) #Learning Rate

        model.compile(optimizer=OPTIMIZER(learning_rate=LEARNING_RATE),loss=LOSS,metrics=METRICS)
        my_bar.progress(25)
        
        X_train = np.loadtxt(train_in_file,skiprows=1,delimiter=',')
        Y_train = np.loadtxt(train_out_file,skiprows=1,delimiter=',')
        X_valid = np.loadtxt(test_in_file,skiprows=1,delimiter=',')
        Y_valid = np.loadtxt(test_out_file,skiprows=1,delimiter=',')
        BATCH_SIZE=int(batch_in)
        EPOCHS=int(epochs_in)
        my_bar.progress(50)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
        #st.write("####setup#####")
        #st.write("TrainingData_INPUT",train_in_file)
        #st.write("TrainingData_OUTPUT",train_out_file)
        #st.write("TestData_INPUT",test_in_file)
        #st.write("TestData_OUTPUT",test_out_file)
        #st.write("Layer Number",5)
        #st.write("BATCH",BATCH_SIZE)
        #st.write("EPOCH",EPOCHS)
        #st.write("Learning Rate",LEARNING_RATE)
        #st.write("####setup#####")
        
        #cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #cp_callback = tf.keras.callbacks.Mode(
         #   filepath=self.target_file5, 
         #   save_weights_only=True)
        my_bar.progress(75)
        with st.spinner('Wait for it...'):
            hist = model.fit(x=X_train,
                            y=Y_train,
                            validation_data=(X_valid,Y_valid),
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            #callbacks=[es,cp_callback]
                            #callbacks=[es]
                            )
        st.success('Done!')

        my_bar.progress(90)
        savepath=workdir+'my_DNN_model'
        model.save(savepath)
        my_bar.progress(100)

        train_loss = hist.history['loss']
        np.savetxt("train_loss.csv",train_loss,delimiter=',')
        #st.write("debug",train_loss)
        valid_loss = hist.history['val_loss']
        #np.savetxt("valid_loss.csv",valid_loss,delimiter=',')
        #st.write("debug",valid_loss)
        epochs = len(train_loss)
        plotx=range(epochs)        
        #np.savetxt("plotx.csv",plotx,delimiter=',')
        l=np.array([train_loss,valid_loss])
        l=l.T
        ll=pd.DataFrame(
          l,columns=["train_loss","valid_loss"])
        #np.savetxt("plotl.csv",l,delimiter=',')
        plotdata=pd.DataFrame(ll)
        st.line_chart(plotdata,use_container_width=True)
        return savepath
#==============================================================================
def DNN_pred():
    savepath=workdir+'my_DNN_model'
    newmodel=tf.keras.models.load_model(savepath)
    predx=np.loadtxt(pred_in_file,skiprows=0,delimiter=',')
    predy=newmodel.predict(predx)
    savepredy=workdir+'pred_y.csv'
    np.savetxt(savepredy,predy,delimiter=',',)
    if yorn == 'yes':
        y_true=np.loadtxt(true_out_file,skiprows=0,delimiter=',')
        savetruey=workdir+'true_y.csv'
        np.savetxt(savetruey,y_true,delimiter=',',)
        rmse = mean_squared_error(y_true,predy, squared=False)
        st.write("R^2=",rmse)
        df_a=pd.read_csv("true_y.csv", header=None, names=["true"])
        df_b=pd.read_csv("pred_y.csv", header=None, names=["pred"])
        #st.write(df_b)
        scatter = alt.Chart(df_a.merge(df_b, left_index=True, right_index=True)).mark_point().encode(
            x=alt.X('true', scale=alt.Scale(domain=(0, 1))),
            y=alt.Y('pred', scale=alt.Scale(domain=(0, 1)))
        )
        line = pd.DataFrame({'true': [0, 1], 'pred': [0, 1]})
        line_chart = alt.Chart(line).mark_line(color='red').encode(
            x=alt.X('true', scale=alt.Scale(domain=(0, 1))),
            y=alt.Y('pred', scale=alt.Scale(domain=(0, 1)))
        )
        chart = scatter + line_chart
    st.altair_chart(chart, use_container_width=True)
#==============================================================================

#image=Image.open("C:\\Users\\ThinkPad\\Desktop\\s：Streamlit\\DNN.jpeg")
with st.sidebar:
    st.header("file path")
    train_in_file=st.file_uploader("Upload Training INPUT Data",type=(["csv"]))
    train_out_file=st.file_uploader("Upload Training OUTPUT Data",type=(["csv"]))
    test_in_file=st.file_uploader("Upload Test INPUT Data",type=(["csv"]))
    test_out_file=st.file_uploader("Upload Test OUTPUT Data",type=(["csv"]))
    pred_in_file=st.file_uploader("Upload Predict INPUT Data",type=(["csv"]))
    true_out_file=st.file_uploader("Upload True_out INPUT Data",type=(["csv"]))


tab1,tab2,tab3,tab4=st.tabs(["DNN setting","DNN check","Training","Predict"])
with tab1:
    col1,col2=st.columns(2)
    with col1:
      st.header("Neural Network Setting")
      #st.image(image, caption='Neural Network')
      workdir=st.text_input("Please input work directry",'.\\')
      batch_in=st.number_input("Please input BATCH",10,2000,10,step=10)
      epochs_in=st.number_input("Please input EPOCHS",10,2000,100,step=10)
    with col2:
      in_num=st.number_input("Please input parameter number",1,2000,8,step=1)
      acti1 = st.selectbox(
          'select activation1',
          ('ReLU','tanh','softmax','linear', 'swish', 'Mish'))
      hid1_unit_num=st.number_input("Please input units number of Hidden Layer 1 ",1,2000,32,step=1)
      acti2 = st.selectbox(
          'select activation2',
          ('ReLU','tanh','softmax','linear', 'swish', 'Mish'))
      hid2_unit_num=st.number_input("Please input units number of Hidden Layer 2 ",1,2000,16,step=1)
      acti3 = st.selectbox(
          'select activation3',
          ('ReLU','tanh','softmax','linear', 'swish', 'Mish'))
      hid3_unit_num=st.number_input("Please input units number of Hidden Layer 3 ",1,2000,8,step=1)
      acti4 = st.selectbox(
          'select activation4',
          ('ReLU','tanh','softmax','linear', 'swish', 'Mish'))
      hid4_unit_num=st.number_input("Please input units number of Hidden Layer 4 ",1,2000,4,step=1)
      acti5 = st.selectbox(
          'select activation5',
           ('ReLU','tanh','softmax','linear', 'swish', 'Mish'))
      hid5_unit_num=st.number_input("Please input units number of Hidden Layer 5 ",1,2000,2,step=1)
      acti6 = st.selectbox(
          'select activation final',
          ('linear','tanh','softmax','ReLU', 'swish', 'Mish'))
      out_num=st.number_input("Please output parameter number",1,2000,1,step=1)

with tab2:
    col1,col2=st.columns(2)
    with col1:
        st.write("INPUT",in_num)
        st.write("activation",acti1)
        st.write("Layer1 Units",hid1_unit_num)
        st.write("activation",acti2)
        st.write("Layer2 Units",hid2_unit_num)
        st.write("activation",acti3)
        st.write("Layer3 Units",hid3_unit_num)
        st.write("activation",acti4)
        st.write("Layer4 Units",hid4_unit_num)
        st.write("activation",acti5)
        st.write("Layer5 Units",hid5_unit_num)
        st.write("activation",acti6)
        st.write("OUTPUT",out_num)
    with col2:
        st.write("Please Check Your selected!!")
        st.write("EPOCH",epochs_in)
        st.write("BATCH_SIZE",batch_in)

with tab3:
    if st.button('EXE'):
        st.write('execute DNN Training!!')
        DNN_func()
    else:
        st.write('waiting')
with tab4:
    yorn = st.selectbox(
          'select R^2 caluclation',
          ('yes','no'))
    if st.button('Pred EXE'):
        st.write('execute Predict!!')
        DNN_pred()
    else:
        st.write('waiting')
