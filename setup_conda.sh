# Download miniconda for python 3
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install conda
bash miniconda.sh -b -p $HOME/customconda

# Configure miniconda
$HOME/customconda/bin/conda config --set always_yes yes --set changeps1 no
$HOME/customconda/bin/conda config --add channels conda-forge
$HOME/customconda/bin/conda update conda

# Custom environments
$HOME/customconda/bin/conda create --name handwriting python
source $HOME/customconda/bin/activate handwriting

# jupyter
$HOME/customconda/bin/conda install jupyter
# Required libraries
$HOME/customconda/bin/conda install colorama==0.3.9
$HOME/customconda/bin/conda install tqdm==4.17.1
$HOME/customconda/bin/conda install pandas==0.20.3
$HOME/customconda/bin/conda install matplotlib==2.0.2
$HOME/customconda/bin/conda install pytest==3.2.1
$HOME/customconda/bin/conda install pytest-suger==0.9.0
$HOME/customconda/bin/conda install numpy==1.13.3
$HOME/customconda/bin/conda install scikit-learn==0.19.1
# Use pip for sphinx to avoid package issues
$HOME/customconda/bin/pip install sphinx==1.6.5
$HOME/customconda/bin/pip install sphinx-autobuild==0.7.1
$HOME/customconda/bin/pip install sphinxcontrib-napoleon==0.3.1
$HOME/customconda/bin/pip install sphinx_rtd_theme==0.2.4
# Install torch on cpu or gpu if we find cuda 8
CUDASTR=$(nvcc --version)
SMISTR=$(nvidia-smi)
CUDAVERSION="8.0"
if echo "$CUDASTR" | grep -q "$CUDAVERSION"; then

	if grep -q "failed" <<< "$SMISTR"; then
		echo "CUDA not found: Install CPU pytorch"
		$HOME/customconda/bin/conda install pytorch==0.2.0 -c soumith;
	else
		echo "CUDA found : Install GPU pytorch"
		$HOME/customconda/bin/conda install pytorch==0.2.0 cuda80 -c soumith;
	fi
else
	echo "CUDA not found: Install CPU pytorch"
	$HOME/customconda/bin/conda install pytorch==0.2.0 -c soumith;
fi
