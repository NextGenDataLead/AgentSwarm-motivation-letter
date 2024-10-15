#Deriving the latest base image
FROM python:latest

#Labels as key value pair
LABEL Maintainer="ampck"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /home/ampck/code/python/ai_motivation_letter/AgentSwarm-motivation-letter/

#to COPY the remote file at working directory in container
COPY / ./
# Now the structure looks like this '/usr/app/src/test.py'

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python", "./Motivation_letter_fastAPI2.py"]
