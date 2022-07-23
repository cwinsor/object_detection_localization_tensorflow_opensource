where I'm at...
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
INSTALLATION
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation
--->with specific notes below <---

create conda env, install TensorFlow:
  conda create -n tensorflow pip python=3.9
  conda activate tensorflow
  pip install --ignore-installed --upgrade tensorflow==2.5.0
verify your installation:
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
<gpu support - I skipped this>
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
    pip install protobuf==3.20
  <the steps that did not work were...>
    get protobuf .zip from https://github.com/google/protobuf/releases
    <note I'm on an AWS Workspace host... so I put that at D:\
    unzip resulting in folder D:\protoc-21.3-win64
    Add <PATH_TO_PB>\bin to your Path environment variable
    open a new PowerShell
    # From within TensorFlow/models/research/
    protoc object_detection/protos/*.proto --python_out=.
COCO Installation
  pip install cython
  pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
  Install (or confirm that it is already installed) Visual C++ 2015 build tools must be installed and on your path.
  get that at https://go.microsoft.com/fwlink/?LinkId=691126 or at https://visualstudio.microsoft.com/visual-cpp-build-tools/
  (you need to 'check' the Visual Studio checkbox...)
Installation of the Object Detection API
  # From within TensorFlow/models/research/
  cp object_detection/packages/tf2/setup.py .
  python -m pip install --use-feature=2020-resolver .
Test the installation
  # From within TensorFlow/models/research/
  python object_detection/builders/model_builder_tf2_test.py

-------------------------------------------
CREATING THE CUSTOM OBJECT DETECTOR
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

Preparing the Dataset
  Install LabelImg
    pip install labelImg

Gather images (100 to 500)
use labelImg to label
copy into test/train 90/10
Goal is create "tf records"
setup configuration file to train

somehelpercode from racoon object detection "dat tran" "xml-to-csv"
https://www.youtube.com/watch?v=W0sRoho8COI
https://github.com/datitran/raccoon_dataset
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

back to the tensorflow instructions...
here are the folders/filer shown in the above tree
  annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.
  exported-models: This folder will be used to store exported versions of our trained model(s).
  images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.
  images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.
  images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.
  models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.
  pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.
  README.md: This is an optional file which provides some general information regarding the training conditions of our model. It is not used by TensorFlow in any way, but it generally helps when you have a few training folders and/or you are revisiting a trained model after some time.

Partition 90/10 into images/train images/test

Creating the Label Map
TensorFlow requires a Label Map
