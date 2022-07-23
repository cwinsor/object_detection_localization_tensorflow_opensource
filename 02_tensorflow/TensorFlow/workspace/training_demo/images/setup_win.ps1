################################################
# activate the virtual environment
# if the pymote_env does not exist - create it

if (-NOT(Test-Path '.\venv\Scripts\activate' -PathType Leaf)) {

    " "
    "did not find venv - creating it now..."

    # confirm at least we have python with pip
    python --version
    python -m pip --version

    # create an empty virtualenv
    python -m venv venv
    }

# activate the venv
.\venv\Scripts\activate

# ensure pip, setuptools and wheels are up to date

Write-Output "upgrade pip"
python -m pip install --upgrade pip
Write-Output "upgrade setuptools"
python -m pip install --upgrade setuptools
Write-Output "upgrade wheel"
python -m pip install --upgrade wheel
Write-Output "upgrade wheel(done)"

# get the libraries specified in requirements.txt and show a list
python -m pip install -r requirements.txt
python -m pip list

################################################
# add to PYTHONPATH for python modules
$Env:PYTHONPATH= '.'
$Env:PYTHONPATH= $Env:PYTHONPATH + ';' + $PSScriptRoot + '\lib;'

Write-Output("setup complete")
