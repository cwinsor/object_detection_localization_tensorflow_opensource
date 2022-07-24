
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

The example uses:

TensorFlow Object Detection API
TensorFlow Detection Model Zoo - pre-trained models. Used as a seed for custom model. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
pycocotools... COCO (Common Object in Context) is a set of high quality datasets for vision.  300K images, 200K labeled, 1.5M object instances, 80 object categories
PROTOBUF - serializing structured data from Google - define how you want your data to be strucured once - then can use speicalized source code to read/write (all languages)
labelImg - labeling tool https://github.com/heartexlabs/labelImg

Also see "few shot training colab" - https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb

Good tutorial video
https://www.youtube.com/watch?v=K_mFnvzyLvc&t=553s
paperspace

-------------------------------------------
JUST RUN EXISTING
cd D:\Users\CHRIS\Documents\code_obj_det_tf_open\object_detection_localization_tensorflow_opensource\02_tensorflow\TensorFlow\workspace\training_demo
conda activate tensorflow
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config


-------------------------------------------
INSTALLATION (every time marked with "********************")
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation
with additional specific notes below

If existing environment...


Create new environment...
create conda env, install TensorFlow:
  conda create -n tensorflow pip python=3.9

  conda activate tensorflow   *********************************++
  pip install --ignore-installed --upgrade tensorflow==2.5.0
  pip install --ignore-installed --upgrade tensorflow==2.5.0

verify your installation (before CUDA)
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"    *********************

GPU Support...

  Install CUDA Toolkit
    Follow this link to download and install CUDA Toolkit 11.2
    Installation instructions can be found here
    !!! on AWS Workspace - DO NOT install display drivers !!!  just cuda toolkit
    !!! on AWS Workspace - DO NOT install to C: rather create folders on D: and install there

  Install CUDNN
    extract the zip file - and copy "cuda" under the toolkit/cuda/11 folder

  Add PATHs to environment PATH variable
    <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
    <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
    <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include
    <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64
    <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\cuda\bin

D:\Users\CHRIS\Anaconda3\envs\tensorflow;D:\Users\CHRIS\Anaconda3\envs\tensorflow\Library\mingw-w64\bin;D:\Users\CHRIS\Anaconda3\envs\tensorflow\Library\usr\bin;D:\Users\CHRIS\Anaconda3\envs\tensorflow\Library\bin;D:\Users\CHRIS\Anaconda3\envs\tensorflow\Scripts;D:\Users\CHRIS\Anaconda3\envs\tensorflow\bin;D:\Users\CHRIS\Anaconda3\condabin;D:\cuda_toolkit\bin;D:\cuda_toolkit\libnvvp;D:\cuda_toolkit\include;D:\cuda_toolkit\extras\CUPTI\lib64;D:\cuda_toolkit\cuda\bin;D:\protoc-21.3-win64\bin;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0;C:\Windows\System32\OpenSSH;C:\Program Files\Amazon\cfn-bootstrap;C:\Program Files\Git\cmd;C:\Program Files\NVIDIA Corporation\Nsight Compute 2020.3.1;D:\Users\CHRIS\AppData\Local\Microsoft\WindowsApps;.;D:\Users\CHRIS\AppData\Local\Programs\Microsoft VS Code\bin

GOT:  error: Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice
FIX: https://stackoverflow.com/questions/72499414/i-got-an-error-about-error-cant-find-libdevice-directory-cuda-dir-nvvm-libd
I had the same problem and just fixed it. The library can't find the folder even if you set the "CUDA_DIR" because it's not using that variable or any other I tried. This post is helpful in understanding the issue. The only solution I was able to find is just copying the required files.
Steps for a quick fix:
Find where your CUDA nvvm is installed (for me it is "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6").
Find the working directory for your script (the environment or the directory you are running the script in).
Copy the entire nvvm folder into the working directory and your script should work.
This is not a great solution but until someone else posts a answer you can at least run your code.





verify your installation (after CUDA)
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"    *********************++

TensorFlow Object Detection API Installation:

Create a new folder under a path of your choice and name it TensorFlow
  cd TensorFlow
  git clone https://github.com/tensorflow/models
  <note - currently at b8234e657. The history is...
* b8234e657 (HEAD -> master, origin/master, origin/HEAD) Fix dropout rate bug.
* a5a652280 Add additional parameters for processing different image shape and label type.
* 754debb0f Added a link to the model-garden guide (#10713)
* ad8470365 Added link to the model-garden guide
* 8d710ccf7 Added model-garden guide published on TF website
* 7c917ef24 Register ViT architecture hyperparams.
*   ddc9bce08 Merge pull request #10626 from ryan0507:master
|\
| * e3fc61e71 Minor bug fixed with yt8m_input_test while refactoring name, Changed class names
| * 83ae348ba eval_utils unittest, typo bug fixed
| *   aef943ed8 Merge branch 'tensorflow:master' into master
| |\
| * | 67ad909de YT8M config added, unused util function deleted
* | | 4323d37cc Add the option of using "num_additional_channels" to ssd_mobilenet_v2_fpn_keras.

You should now have a single folder named models under your TensorFlow folder, wh
Install protobuf
  <hmm - the protobuf did not work in the end.  I ended up coming back to this step at the end and using...>
    pip uninstall protobuf
   conda  install protobuf
    which gave me protobuf-3.20.1
  <the following did not work...>
    pip install protobuf<3.20    
    pip install protobuf==3.20 <- this was incompitible
  <other steps that did not work were...>
    get protobuf .zip from https://github.com/google/protobuf/releases
    <note I'm on an AWS Workspace host... so I put that at D:\
    unzip resulting in folder D:\protoc-21.3-win64
    Add <PATH_TO_PB>\bin to your Path environment variable
    open a new PowerShell
    # From within TensorFlow/models/research/
    protoc object_detection/protos/*.proto --python_out=.      *********************
    OR - note the "Important" in the site...
    Get-ChildItem object_detection/protos/*.proto | foreach {protoc "object_detection/protos/$($_.Name)" --python_out=.}

COCO Installation
  pip install cython    *********************
  pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI    *********************
  Install (or confirm that it is already installed) Visual C++ 2015 build tools must be installed and on your path.
  get that at https://go.microsoft.com/fwlink/?LinkId=691126 or at https://visualstudio.microsoft.com/visual-cpp-build-tools/
  (you need to 'check' the Visual Studio checkbox...)
Installation of the Object Detection API
  # From within TensorFlow/models/research/
  cp object_detection/packages/tf2/setup.py .
  python -m pip install --use-feature=2020-resolver .   ********************

Test the installation
  # From within TensorFlow/models/research/            ********************
  python object_detection/builders/model_builder_tf2_test.py     ********************

-------------------------------------------
CREATING THE CUSTOM OBJECT DETECTOR
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

The goal is TensorFlow "records" file by labeling objects in the image.

somehelpercode from racoon object detection "dat tran" "xml-to-csv"
https://www.youtube.com/watch?v=W0sRoho8COI
https://github.com/datitran/raccoon_dataset
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

back to the tensorflow instructions...
here are the folders/files
  annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.
  exported-models: This folder will be used to store exported versions of our trained model(s).
  images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.
  images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.
  images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.
  models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.
  pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.
  README.md: This is an optional file which provides some general information regarding the training conditions of our model. It is not used by TensorFlow in any way, but it generally helps when you have a few training folders and/or you are revisiting a trained model after some time.

PREPARING THE DATASET
  Install LabelImg
    pip install labelImg

Gather images (100 to 500)
Convert from .tif to .png
use to_png.py and setup_win.ps1 in workspace\training_demo\images

Annotate Images
  use labelImg for this
  You will thus have .png and .xml

Partition the Dataset
  Partition 90/10 and copy .png and .xml into images/train images/test

Create the Label Map
  TensorFlow requires a Label Map.  This is a stupid file which maps between numeric class values and 'friendly' name
  Label map files have the extention .pbtxt and should be placed inside the training_demo/annotations folder.

Create TensorFlow Records
  we are converting .xml to .record  (object locations)
  cd into TensorFlow/scripts/preprocessing and run:
  scripts> python generate_tfrecord.py -x ..\workspace\training_demo\images\train\ -l ..\workspace\training_demo\annotations\label_map.pbtxt -o ..\workspace\training_demo\annotations\train.record
  scripts> python generate_tfrecord.py -x ..\workspace\training_demo\images\test\ -l ..\workspace\training_demo\annotations\label_map.pbtxt -o ..\workspace\training_demo\annotations\test.record

CONFIGURE A TRAINING JOB

Download pre-trained model
  get from  TensorFlow 2 Detection Model Zoo (e.g.SSD ResNet50 V1 FPN 640x640)
  extract ito training_demo/pre-trained-models

Configure training pipeline
  Under the training_demo/models create a new directory named my_ssd_resnet50_v1_fpn
  and copy the training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config file
  Edits per the instructions...

Training the model
  Copy the TensorFlow/models/research/object_detection/model_main_tf2.py script 
  and paste it straight into \TensorFlow\workspace\training_demo
  cd 02_tensorflow\TensorFlow\workspace\training_demo  and run
  *******************+++
  python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
  IF PYTHON CRASHES ... probably due to low memory !!!  need >= 64GB