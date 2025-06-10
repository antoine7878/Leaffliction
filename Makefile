COMPOSE_FILE = ./docker-compose.yaml
RUN_DOCKER = docker exec -it tensorflow

all: up

up:
	docker compose --file=$(COMPOSE_FILE) up -d

build:
	docker compose --file=$(COMPOSE_FILE) up -d --build

stop:
	docker compose --file=$(COMPOSE_FILE) stop -t 0

exec:
	$(RUN_DOCKER) bash

dclean:
	docker ps -aq | xargs docker rm -f ;\
	docker image ls -q | xargs docker image rm -f ;\
	docker builder prune -f ;\

distribution:
	$(RUN_DOCKER) python src/Distribution.py /goinfre/images

augmentation:
	time python Augmentation.py ~/goinfre/images

clean:
	time python Augmentation.py --clean ~/goinfre/images/

transformation:
	time python Transformation.py ~/goinfre/images/Apple_Black_rot/image\ \(100\).JPG

train:
	time python ./train.py ~/goinfre/images

predict:
	time python predict.py ~/goinfre/Leaffliction.zip

.PHONY: train eval
