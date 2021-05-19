.PHONY: build run

build: 
	docker build . -t image-segmentation-unet:0.1

run:
	docker run --name unet -v ${PWD}/results/:/home/image_segmentation/results/ -v ${PWD}/logs/:/home/image_segmentation/logs/ image-segmentation-unet:0.1


# TODO: Add other targets, for exmaple config file check or testing
#check: 
#	docker run --name unet_check image-segmentation-unet:0.1 config_checking.py test_config_funct