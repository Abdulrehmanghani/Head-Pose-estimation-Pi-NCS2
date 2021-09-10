# Accelerate the Inference on the Edge

Computer vision is not just affiliated with breaking down the images and videos into pixels, but also making these pixels to represent a class. There are a lot of Computer vision models developed in TensorFlow for object detection and image segmentation. Recent developments in deep learning have observed remarkable performance in the AI industry with highly accurate computer vision models and it is observed that computational power is getting cheaper. But data-driven decisions in deep learning and cloud computing based systems have limitations in deployment at edge devices in real-world scenarios. Since we cannot bring edge devices to the data-centers, We can't deploy a server/GPU at the edge where we can deploy our applications at low cost embedded devices, so we need to bring AI to the edge on low cost embedded devices.

Intel distribution of openVINO toolkit is a powerful tool, which provides the optimization techniques to optimize the computer vision and non-computer vision  models into intermediate representation that will allow us to deploy these models at edge on low cost embedded devices. Open vino supports different deep learning frameworks (tensorflow, ONNX,caffe,etc) it provides optimization techniques to optimize pre-train models and convert them into IR representation, which is used to accelerate the inference at the edge. Intermediate representation contains two files xml and bin. Xml files contain the architecture of the model and bin files contain the weights of the model.

Training and optimization of the model must be done on a GPU or a high end server. Then this trained model fed to the model optimizer to generate the IR format. The models are optimized by using different techniques like quantization, freezing, fusion, pruning and more. Optimization is done by using a single command provided by openvino based on the chosen framework. We also can choose models from Openvino model zoo; these are pretrain and in the IR format. Openvino model zoo contain every kind of model of classification, object detection, re-identification and image segmentation. These optimized models can be downloaded by using a single command provided by openvnio. We can download it with different quantizations FP32,FP16 and INT8.  

Now this IR format of the model is fed to the inference engine of the openvino. Inference engine will check the model compatibility with chosen framework as well as with the Hardware to run on. Inference engine provides the hardware plugins to run the optimized model. We will choose the plugins at run time for that hardware which we are using to deploy for example, in case of Neural Compute Stick 2 we will pass MYRIAD as an argument to select the plugins for NCS2.
Openvino optimization tools also reduce the model footprint. Footprint refers to the space and latency of the model. The amount of space and latency of the model needs to be low at the edge because we have limited resources at the edge in terms of computational power, memory size. 

Openvino provides the Multi-Device plugin feature which automatically assigns inference requests to available computational devices (CPU/IGPU/VPU) to execute the requests in parallel. Multi-device plugins will improve the throughput of the system which has multiple devices integrated on it. Because of the different integrated devices will share the workload if one device is busy then another device can take more of the load. In this case we don't need to load the models explicitly on every device we need to create a balanced inference request. If we are processing 4 cameras on the CPU and we want to process 4 inference requests we need to utilize CPU and GPU via multi-device plugins.

# Use case
Openvino toolkit is built to felicitate the highly accurate models of computer vision and deep learning on the different Intel plat-forms. It can be used in the remote surveillance, retail, security, health care, transport and many more.

# FPS comparison:
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model name</th>
      <th>CPU</th>
      <th>CPU+NCS2</th>
      <th>Raspberry Pi(Desktop OS) + NCS2</th>
      <th>Raspberry Pi(Headless OS) + NCS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SSD MobileNet v1</td>
      <td>23.4562</td>
      <td>10.6245</td>
      <td>6.0412</td> 
      <td>6.934</td> 
    </tr>
  </tbody>    
  <tbody>
    <tr>
      <th>1</th>
      <td>SSD MobileNet v2</td>
      <td>28.0894</td>
      <td>8.9829</td>
      <td>5.4806</td> 
      <td>6.3106</td> 
    </tr>
  </tbody>  
  <tbody>
    <tr>
      <th>2</th>
      <td>Tiny YOLO v3</td>
      <td>23.4288</td>
      <td>7.96264</td>
      <td>3.8875</td> 
      <td>--</td> 
    </tr>
  </tbody>  
</table>

# Model Load Time:
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model name</th>
      <th>CPU</th>
      <th>CPU+NCS2</th>
      <th>Raspberry Pi(Desktop OS) + NCS2</th>
      <th>Raspberry Pi(Headless OS) + NCS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SSD MobileNet v1</td>
      <td>0.1097</td>
      <td>2.6432</td>
      <td>9.8460</td> 
      <td>9.910</td> 
    </tr>
  </tbody>    
  <tbody>
    <tr>
      <th>1</th>
      <td>SSD MobileNet v2</td>
      <td>0.1256</td>
      <td>2.9364</td>
      <td>15.64</td> 
      <td>16.0970</td> 
    </tr>
  </tbody>  
  <tbody>
    <tr>
      <th>2</th>
      <td>Tiny YOLO v3</td>
      <td>0.0768</td>
      <td>2.3918</td>
      <td>5.6864</td> 
      <td>--</td> 
    </tr>
  </tbody>  
</table>

