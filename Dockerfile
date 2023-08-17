# Use preconfigured official pytorch image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install libgomp
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install --no-install-recommends -y libegl1 libgl1 libgomp1


# Copy repository files to image directory
COPY . /home/OCTA-seg/

# Install dependencies
RUN pip install -r /home/airway-seg/requirements.txt

# RUN chmod 755 /home/airway-seg/dockershell.sh
RUN echo "Successfully build image!"

ENTRYPOINT ["/home/OCTA-seg/docker/dockershell.sh"]
