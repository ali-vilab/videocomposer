# sh into dockerfile


# first, build
docker build -t ml-interface . -f dockerfiles/Dockerfile.base

# then, sh into it, with /.cache/ mounted as well as current directory 
docker run -it --rm --name ml-interface -v $(pwd):/app -v ~/.cache/:/root/.cache/ --gpus all ml-interface /bin/bash 
