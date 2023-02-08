# DNEG-ML-Template
Template for projects using DNEG ML Toolkit 

## Installation
DNEG-ML-Template extends the DNEG-ML-Toolkit, following its structure and extending its functionality.
DNEG-ML-Toolkit currently needs to be run from source, which can be found at:
https://github.com/dneg/DNEG-ML-Toolkit/tree/v1.0. Follow the instructions there to set up the environment and install
the Toolkit.

Once this is done, install the following in the same conda environment to run DNEG-ML-Template:
1. Navigate to the root folder of DNEG-ML-Template and run *pip install -r requirements.txt*
2. Install the correct version of PyTorch 1.12 from https://pytorch.org/get-started/previous-versions/#v1121, using the version with the correct cudatoolkit (i.e. 11.3)

### Make Experiment
Make Experiment is a helper tool to speed up creating a new experiment. It allows you to choose a configuration template
to start from, and builds the experiment's folder using this template.

Example usage: 

*python run_experiment.py make-experiment --name {experiment_name} --template Simple*

- *template* is the name of a json configuration file found in the config_templates folder.
- Once the experiment has been created, edit the experiment's json configuration file to provision the environment and set the hyperparameters for the experiment.

### Run Experiment
Run Experiment runs the training of the ML system, using the json configuration for the specified configuration. The experiment
folder and json configuration can be created manually, or you can use the Make Experiment tool to assist with this.

Example usage: 

*python run_experiment.py train --experiment {experiment_name} --device 1*

- *experiment* is the name of the experiment within the experiments folder
- *device* allows the pose and landmarking tools to be run on gpu. 1 will run on 1 gpu, [0] will run on gpu with index 0, cpu will run the training on cpu.
- *resume* Attempt to resume the training from the latest checkpoint. If no checkpoint is found, will start from the beginning.
- *resume_from_checkpoint* Path to a specific checkpoint to resume the training from. Cannot be use if --resume is enabled.