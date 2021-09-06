# Accelerate the Inference on the Edge
Computer vision is not just affiliated with breaking down the images and videos into pixels, but also making these pixels to represent a class. There are a lot of Computer vision models developed in TensorFlow for object detections and image segmentations. Recent developments in deep learning have observed that the computational power is getting cheaper. But data-driven decsion's in deep learning and cloud computing based systems have limitations at edge devices in real-world scenarios. Since we cannot bring edge devices to the data-centers, We can't deploy a server/GPU at the edge where we can deploy our applications at low cost embedded devices. so we bring AI to the edge devices with AI on the Edge.


Intel openVINO toolkit is a powerful tool, which provides the optimization techniques to optimize the computer vision models into intermediate representation that will allow us to deploy these models at edge on low cost embedded devices.

The aim of this article is to deploy the pre-train computer vision models of TensorFlow on raspberry pi with Intel neural compute stick 2. This optimized model can be deployed at the edge in the retail sector for people detection and counting in the public streets, where the government locks down due to the covid-19.
We can deploy these models without extra hardware requirements. This will save memory cost and power consumption as well.


