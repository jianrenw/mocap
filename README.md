conda update -n base -c defaults conda
conda create -n pytorch3d python=3.10 && conda activate pytorch3d
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d::pytorch3d