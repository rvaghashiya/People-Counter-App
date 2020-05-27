# Project Write-Up


## Model Selection and Conversion via Model Optimizer

Object Detection Model Selected : 

An object-detection TensorFlow model named [SSD MobileNet V2 COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz).

To obtain xml and bin files, the model was converted via Model Optimizer using following command line argument:

`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`


## Explaining Custom Layers

All the layers used by this object detection model are supported by the Model Optimizer, hence does not involve handling/support for any custom layer. 

But, if the need arises, a custom layer can be handled in following manner:

	- Register the custom layer as an extension to Model Optimizer, in case of Tensorflow or Caffe models
	  For this purpose, we can either :
		- directly use an available extension, if any, 
		- generate required extension template files via Model Extension Generator, which are needed by the Inference Engine(IE) to execute that custom layer
			The steps for generating the same are as follows, for TensorFlow models:
				- generate required template files for Model Optimizer TensorFlow extractor, Model Optimizer custom layer operation, IE CPU extension (and GPU extension), depending on the hardware being used
				- generate IR files having custom layer using the TF extractor and operation extension with Model Optimizer
				- implement Custom layer for corresponding hardware, say CPU, on which the model is to be run
				- execute the model
	- Register the layer as Custom and calculate the output shape, if using Caffe
	- In TensorFlow, either replace the unsupported subgraph(operation flow) with a different supported one , or offload that computation segment to TensorFlow

Some of the potential reasons for handling custom layers are:

Model Optimizer generally supports those layers which are widely used for inference, across different frameworks.
All the types of functionality provided in the native framework are not necessarily supported by the Model Optimizer, hence we need special handle mechanism to support these custom layers. 

These layers are typically only needed for special and task-specific operation in a given framework. Hence, one of the ways to handle such is to offload computation to the native framework, or use some extension to support these operations.
If not done so, it would lead to breakage in the flow of the model, rendering it unusable for deployment at the network edge.


## Comparing Model Performance

My methods to compare models before and after conversion to Intermediate Representations were...

The size difference of the model pre- and post-conversion :

	- The size of the model pre-conversion  : 66.4 MB
	- The size of the model post-conversion : 64.1 + 0.256 = 64.356 MB

The inference time of the model pre- and post-conversion :

	- The inference time of the model pre-conversion  : 100 ms
	- The inference time of the model post-conversion : 68 ms (nearly 1.5x faster)


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

	- Mass Surveillance in a pandemic: The best use-case of this app can be to ensure social distancing and well as avoid overcrowding during this COVID-19 pandemic. The app can be easily expanded to focus on these scenarios,
		and can be deployed at various public and crowded places like shopping malls, stores, parks, cafes, educational institutes, etc.
	
	- Security Surveillance at Banks and Airports : App can be deployed to monitor suspicious activities at sensitive areas like banks and airports to guard against theft, etc.
		It can be especially effective with a person identification module.
	
	- Retail Stores: Can be deployed to maintain track of people's shopping schedules and patterns, to avoid overcrowding as well as lack of stock, and better resource management as well as utilization.
	 

## Assess Effects on End User Needs

Since person-detection task is basically a image processing task, all factors such as scene illumination, camera resolution and focal length, etc. can have a significant impact on detection

	- Illumination and Foreground-Background Contrast: Person with dark-colored clothes in a dark or dim-lit background might be difficult to detect, especially if the person is not moving.
		Infact, in the test video itself, the person with dark-blue clothes and black hair almost resembles a blob, hence the model misses out detecting that person in many frames.
		The placement of the camera and its field of view will determine the utility of the model. It must be placed such that the scene to be observed has uniform illumination, and free from maximum background clutter which can hamper the detection such as presence of obstacles, mirror reflections, etc.
		
	- Camera focal length and resolution: The greater the power of the camera lens, the better and clearer will be the image captured, and hence will improve the detection as well as the confidence of detection.
	 Again, the app need will define the same. App deployed at airport and banks needs to be much more accurate than at a lottery kiosk, hence needs input of better quality.

	- Model accuracy: The more diverse and varying the dataset on which the model is trained, the better will be the detection results and accuracy. Therefore, a model trained on dataset with varying illumination conditions, poses, etc. will detect a person with much greater accuracy.
		But, with accuracy comes the trade-off of size, hence an optimum accuracy which would suffice the cause of the app must be decided.
		App deployed at military camps must be more robust to detect an infiltrator than those at a cafe/salon where they just manage the customer queues

All of these factors greatly impact the application, and are directly dependant on the app use-case.


## Model Research

In investigating potential people counter models, I tried out a tensorflow-based light-weight and fast object detection model, trained on COOC dataset.

The following steps helped in successful conversion of the model to its IR form :

  - `wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
  - `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
  - `cd ssd_mobilenet_v2_coco_2018_03_29`
  - `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  
