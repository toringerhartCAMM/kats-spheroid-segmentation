
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import getopt
import sys


def usage ():
	print("Usage: {} -i <image directory>".format(sys.argv[0]))
	sys.exit()


def remove_boarder_contours (contours, size, pixels=100):
	height, width = size
	removed_contour_indexes = set()
	for index,contour in enumerate(contours):
		for point in contour:
			if (((point[0][1] < pixels) or (point[0][1] > height-pixels)) or
			    ((point[0][0] < pixels) or (point[0][0] >  width-pixels))):
				   removed_contour_indexes.add(index)
	for index in sorted(removed_contour_indexes, reverse=True):
		del contours[index]


def process_file (filename):
	image  = cv2.imread(filename)
	output = image.copy()
	height = image.shape[0]
	width  = image.shape[1]
	
	# Convert to grayscale
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# edge detection
	edges = cv2.Canny(grayscale, 50, 100)

    # Dilate
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	dilated = cv2.dilate(edges, kernel, iterations=1)

    # Check if floodFill filled more than 25% of the image
	locations = [   (0, 0), 					# Top left
					(0, height - 1), 			# Top right
					(width - 1, 0),     		# Bottom left
					(width - 1, height - 1) ] 	# Bottom right

	for location in locations:
		filled = dilated.copy()
		flood_mask = np.zeros((grayscale.shape[0]+2, grayscale.shape[1]+2), np.uint8)       
		pixel_count, _, _, _ = cv2.floodFill(filled, flood_mask, location, 255)
		if (pixel_count / image.size) > 0.25:
			break
    
    # Invert the filled image
	filled_inverted = cv2.bitwise_not(filled)
    
    # fill the holes
	filled_holes = dilated | filled_inverted
	
	# Find contours
	tmpimg = filled_holes.copy()
	_, contours, _ = cv2.findContours(tmpimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	
	# Remove contours within 100px of the boarder
	remove_boarder_contours(contours, (height, width))
	
	# Find contour with the largest area
	contour_areas = [ cv2.contourArea(contour) for contour in contours ]
	try:
		max_index = max(range(len(contour_areas)), key=contour_areas.__getitem__)
	except ValueError:
		# Didnt find any contours
		return 0, 0, (0,0,0,0), (0,0,0,0), output
	contour_match = contours[max_index]
	
	# Draw the contours on a copy of the original image
	cv2.drawContours(output, contour_match, -1, (0, 0, 255), 2)
	
	# Create a mask from the contour
	contour_mask = np.zeros(grayscale.shape, np.uint8)
	cv2.drawContours(contour_mask, contour_match, -1, (255), -1)
	
	# Calculate contour stats
	contour_area = cv2.contourArea(contour_match)
	equivalent_diameter = np.sqrt(4*contour_area / np.pi)
	
	# Fill in the mask
	filled_mask = contour_mask.copy()
	flood_mask = np.zeros((grayscale.shape[0]+2, grayscale.shape[1]+2), np.uint8)
	cv2.floodFill(filled_mask, flood_mask, (0, 0), 254)
	filled_mask[ filled_mask ==   0 ] = 255 
	filled_mask[ filled_mask == 254 ] = 0
	
	# Take the average of the area
	mean_intensity_color = cv2.mean(image, mask=filled_mask)
	mean_intensity_grayscale = cv2.mean(grayscale, mask=filled_mask)
	
	# Return stats
	return contour_area, equivalent_diameter, mean_intensity_color, mean_intensity_grayscale, output


def main ():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:v")
    except getopt.GetoptError as err:
        print(err) 
        usage()
    
    image_directory = None
    verbose         = False
    
    for opt, arg in opts:
        if   opt == "-h":
            usage()
        elif opt == "-i":
            image_directory = arg
        elif opt == "-v":
            verbose = True
        else:
            assert False, "unhandled option" 
	
	# Check if data was passed properly
    if image_directory is None:
        usage()
	
	# Check if directory terminates with '/', and append if not
    if image_directory[-1] != '/':
        image_directory = image_directory + '/'

	# Get the list of images
    files = glob.glob(image_directory + "/*.tif")
    total_files = len(files)

    # Test for an empty directory
    if total_files == 0:
        usage()
	
	# Test for a directory named "Segmented Images" and create it if needed
    output_directory = image_directory + "Segmented Images/"
    if not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            assert False, "Cannot create directory: {}".format(output_directory)
	
    # Open an output file for writing
    output_file = output_directory + "Results.csv"
    f = open(output_file, "w")
    f.write("Filename,Diameter,Area,Mean Red,Mean Green,Mean Blue,Mean Grayscale\n")
	
    # Iterate through files and process
    for i,file in enumerate(sorted(files)):
		
        filename = os.path.basename(file)
        name = filename.split('.').pop(0)
		if verbose:
			print("[{:.2f}%] Processing {}... ".format((100 * i / total_files), filename), end='', flush=True)

        # Process the image, write the reults, and save the segmented image
        area, diameter, mean_color, mean_grayscale, output = process_file(file)

        mean_red, mean_blue, mean_green, _ = mean_color
        mean_gray = mean_grayscale[0]

        f.write("{},{},{},{},{},{},{}\n".format(filename, diameter, area, mean_red, mean_blue, mean_green, mean_gray))
        cv2.imwrite(output_directory + name + ".jpg", output)

		if verbose:
			print("Done.", flush=True)


if __name__ == "__main__":
    main()
