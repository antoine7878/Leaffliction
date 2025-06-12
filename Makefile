distribution: goinfre
	python src/Distribution.py ~/goinfre/images

augmentation:
	python src/Augmentation.py ~/goinfre/images

clean:
	python src/Augmentation.py --clean ~/goinfre/images/

transformation:
	python src/Transformation.py ~/goinfre/images/Apple_healthy/image\ \(101\).JPG

train:
	python src/train.py ~/goinfre/images

predict:
	python src/predict.py ~/goinfre/Leaffliction.zip

predict1:
	python src/predict.py ~/goinfre/Leaffliction.zip -f ~/goinfre/images/Apple_Black_rot/image\ \(2\).JPG

predictf:
	python src/predict.py ~/goinfre/Leaffliction.zip -f ~/goinfre/images/Apple_rust/

.PHONY: train eval all clean
