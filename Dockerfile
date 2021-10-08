#Building image from debian
FROM debian

MAINTAINER Jayanth Kumar Narayana <jayanth.kumar@tuta.io>

#Installing necessary programs
RUN apt update && apt-get install -y python3 \
	python3-pip \
	r-base \
	&& pip3 install tmap rpy2 psutil \
	&& R -e "install.packages(c('vegan'), repos='http://cran.rstudio.com/')" \	

ENTRYPOINT ["python3"]
