distribution:
	time python Distribution.py ~/goinfre/images

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
