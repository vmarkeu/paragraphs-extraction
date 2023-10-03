# DIP Asgm Part B

"""
Created on Tue Nov 22 01:11:11 2022

@author: vivien mak
"""

import cv2
import numpy as np
import glob
import os

# uncomment this if want to display histogram projection
# from matplotlib import pyplot as pt

# function to convert image to binary image
def findBinarizedImage(img):
    # convert image to grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # use gaussian blur to smooth the image to avoid noises (if any)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # convert grayscale image to binary image (auto select threshold to produce binary img)
    binarized_image = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    return binarized_image

# function to perform closing on image (dilate first hen erode)
def performClosing(img):
    # create structuring element for dilation
    sE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # rotate the structuring element by 180 degree
    rSE = np.rot90(sE, 2)

    # dilate the image with structuring element
    # this process is to join all the gaps for the blank spaces
    dilated_image = cv2.dilate(img, rSE, iterations = 6)
    
    # erode the image with the same structuring element used for dilation
    # this process is to change the dilated image back to the original size after perform dilation
    eroded_image = cv2.erode(dilated_image, rSE, iterations = 6)
    
    return eroded_image

# function to obtain the start and end index of each column/paragraph
def findStartandEndIndex(histogram_projection):
    # define an empty array
    all_start_index = []
    all_end_index = []
        
    # loop each col/paragraph to find where the col/paragraph start from
    # since it loop from the beginning til the end of the histogram projection
    # therefore, the index and values are loop from the beginning to the end
    # hence, the all start & end index is sorted
    for index, col_or_para in enumerate(histogram_projection):
        # get 1st value after not 0, find start col/paragraph
        if col_or_para != 0:
            # add the index into array
            all_start_index.append(index)
        else:
            # add the index into array
            all_end_index.append(index)
    
    return all_start_index, all_end_index

# function to obtain only start index of a column or paragraph
def findNonConsecutive(array_name):
    # define an empty array
    non_consecutive_array = []
    
    # set the start to the first value (which is the index of image) in the array_name
    start = array_name[0]
    
    # then set index as 0
    index = 0
    
    # for start index only
    if array_name[0] != 0:
        # this is to store the start index of the array_name into the array we want (non_consecutive array)
        non_consecutive_array.append(start)
    
    # then loop through the values in the array_name
    for value in array_name:
        # if the value is same as the start value, the start and index will increase by 1
        if start == value:
            start += 1
            index += 1
            continue

        # then it will append the value into the non_consecutive array
        non_consecutive_array.append(value)
        start = value + 1
        index += 1

    return non_consecutive_array

# function to create directory
def createDirectory(dir_name):
    # get current directory
    current_directory = os.getcwd()
    
    # create directory with the "dir_name" to store extracted columns
    create_directory = os.path.join(current_directory, r'' + dir_name + '')
    
    # create the directory if the directory does not exist
    if not os.path.exists(create_directory):
        # create directory
       os.makedirs(create_directory)
       
    return create_directory

# loop through all the .png file in the "images" folder
for image in glob.glob("images/*.png"):
    # get image name with image format
    image_name_with_format = os.path.basename(image)
    
    # get image name only
    image_name = os.path.splitext(image_name_with_format)[0]

    # read image
    img = cv2.imread(image)
    
    # binarized the original image (become 0 and 1)
    binarized_image = findBinarizedImage(img)
    
    # - - - - - - - - - - IGNORE TABLE - - - - - - - - - -
    
    # find contours of binarized image
    contours = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # find size of image
    [nrow, ncol, nlayer] = img.shape
    
    # create mask to extract ROI (without table)
    mask = np.zeros((nrow, ncol, 3), dtype = np.uint8)
    
    # find the bounding rectangle and draw it for each contour
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 1000:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    
    # bitwise or to get the output without the table
    img_wo_table = cv2.bitwise_or(img, mask)
    
    # binarize the original image (become 0 and 1)
    binarized_image = findBinarizedImage(img_wo_table)
    
    # perform closing to the binarized image
    # dilate the image to join the gaps and erode the dilated image to convert back to original size
    closing_image = performClosing(binarized_image)
    
    # - - - - - - - - - - COLUMN - - - - - - - - - -

    # vertical projection (closing)
    # axis = 0 -> columns
    closing_vertical_projection = np.sum(closing_image, axis = 0)

    # vertical projection -> loop each column to find where the column start from
    all_col_start_index, all_col_end_index = findStartandEndIndex(closing_vertical_projection)
    
    # find start & end index of column
    col_start_index = findNonConsecutive(all_col_start_index)
    col_end_index = findNonConsecutive(all_col_end_index)
        
    # extract column from image
    count_column = 1
    extracted_column_array = []
    for i, j in zip(col_start_index, col_end_index):
        # extract the region out (column)
        # add padding to the left & right of the column to make it not start extract from the text
        extracted_column = img_wo_table[:, i-10:j+10]
        extracted_column_array.append(extracted_column)
        
        # create directory
        column_directory = createDirectory("extracted_columns")
        
        # save extracted column
        cv2.imwrite(str(column_directory) + "/" + str(image_name) + "_column_" + str(count_column) + ".png", extracted_column)
        
        # increment to extract next column
        count_column += 1
        
    # - - - - - - - - - - PARAGRAPH - - - - - - - - - - 
    
    # loop through the length of the extracted_column_array
    for i in range(len(extracted_column_array)):
        # read the column image
        col_name = str(image_name) + "_column_" + str(i+1)
        column = cv2.imread(str(column_directory) + "/" + str(col_name) + ".png")
        
        # binarize the column image (become 0 and 1)
        binarized_column = findBinarizedImage(column)
        
        # perform closing to the binarized column
        # dilate the column to join the gaps and erode the dilated column to convert back to original size
        closing_column = performClosing(binarized_column)

        # horizontal projection (closing)
        # axis = 1 -> rows
        closing_horizontal_projection = np.sum(closing_column, axis = 1)
            
        # horizontal projection -> loop each row to find where the row start from
        all_row_start_index, all_row_end_index = findStartandEndIndex(closing_horizontal_projection)
            
        # find start & end index of paragraph
        row_start_index = findNonConsecutive(all_row_start_index)
        row_end_index = findNonConsecutive(all_row_end_index)
        
        # extract paragraph from column image
        count_row = 1
        for i, j in zip(row_start_index, row_end_index):
            # extract the region out (paragraph)
            # add padding to the top & bottom of the paragraph to make it not start extract from the text
            extracted_row = column[i-10:j+10, :]
            
            # create directory
            paragraph_directory = createDirectory("output")
            
            # save extracted paragraph
            cv2.imwrite(str(paragraph_directory) + "/" + str(col_name) + "_paragraph_" + str(count_row) + ".png", extracted_row)
            
            # increment to extract next paragraph
            count_row += 1
    
        # # - - - - - - - - - - DISPLAY HISTOGRAM PROJECTION - - - - - - - - - -
        
        # # display original image
        # pt.figure()
        # pt.subplot(1 ,3, 1)
        # pt.imshow(img, cmap = "gray")
        # pt.title(image_name)
        
        # # display black pixel count of column (to identify position of column)
        # pt.subplot(1, 3, 2)
        # pt.plot(closing_vertical_projection)
        # pt.title("Vertical Projection")
        # pt.xlabel("Column Number")
        # pt.ylabel("Count")
        
        # # display black pixel count of row (to identify position of paragraph)
        # pt.subplot(1, 3, 3)
        # pt.plot(closing_horizontal_projection)
        # pt.title("Horizontal Projection " + str(col_name))
        # pt.xlabel("Row Number")
        # pt.ylabel("Count")