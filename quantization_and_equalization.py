import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave
from skimage.color import rgb2gray


TRANSFORM_MATRIX = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    This function reads an image and turns it to black and white if wanted.
    :param filename:
    :param representation: 1 for black and white image, otherwise a RGB image.
    :return:
    """
    im = imread(filename) / 255
    if representation == 1:
        im = rgb2gray(im)
    return im


def imdisplay(filename, representation):
    """
    This function displays an image.
    :param filename:
    :param representation:
    :return:
    """
    image = read_image(filename, representation)
    plt.figure()
    if representation == 1:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)
    plt.show()


def rgb2yiq(im_RGB):
    """
    This function turns a RGB image into a YIQ image.
    :param im_RGB:
    :return:
    """
    return np.dot(im_RGB, TRANSFORM_MATRIX.T)


def yiq2rgb(im_YIQ):
    """
    This function turns a YIQ image into a RGB image.
    :param im_YIQ:
    :return:
    """
    return np.dot(im_YIQ, np.linalg.inv(TRANSFORM_MATRIX).T)


def histogram_equalize(im_orig):
    """
    This function equalizes the gray colors of the image, which increases the contrast in the image by using a variety
    of gray colors from black to white.
    :param im_orig:
    :return:
    """
    image_y = im_orig
    if im_orig.ndim == 3:
        yiq = rgb2yiq(im_orig)
        image_y = yiq[:, :, 0]
    hist_orig = np.histogram(image_y, bins=256, range=(0, 1))[0]
    cum_hist = np.cumsum(hist_orig)
    first_non_zero = np.nonzero(cum_hist)[0][0]
    lookup_table = np.clip(
        np.round(255 * (cum_hist - cum_hist[first_non_zero]) /
                 (cum_hist[-1] - cum_hist[first_non_zero])), 0, 255).astype(
        np.uint8)
    int_image_y = np.round(image_y * 255).astype(int)
    new_image_y_int = lookup_table[int_image_y]
    new_image_y_float = new_image_y_int / 255
    img_after_eq = new_image_y_float
    if im_orig.ndim == 3:
        yiq[:, :, 0] = new_image_y_float
        img_after_eq = yiq2rgb(yiq)
    hist_eq = np.histogram(new_image_y_float, bins=256)[0]
    return [img_after_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    This function quantizes the gray colors of the image, which is using less gray colors in the image (combining
    similar gray colors to one color and keeping the image as close as possible to the original).
    :param im_orig:
    :param n_quant:
    :param n_iter:
    :return:
    """
    # assumimg image is black and white
    image_y = im_orig
    if im_orig.ndim == 3:
        yiq_image = rgb2yiq(im_orig)
        image_y = yiq_image[:, :, 0]
    int_image = np.round(image_y * 255).astype(np.uint8)
    hist_image = np.histogram(int_image, bins=256, range=(0, 255))[0]
    cum_hist = np.cumsum(hist_image)

    p_hist_image = hist_image / int_image.size

    # calculate first z values
    z_values = [-1]
    for i in range(1, n_quant):
        z_values.append(np.argmin(cum_hist < i * int(int_image.size /
                                                     n_quant)))
    z_values.append(255)
    z_values = np.array(z_values)
    q_values = np.zeros((n_quant,))
    all_zs = np.arange(256)
    # calculate first q values
    for z in range(1, len(z_values)):
        chunk_of_p = p_hist_image[z_values[z - 1] + 1:z_values[z] + 1]
        z_chunk = all_zs[z_values[z - 1] + 1:z_values[z] + 1]
        q_values[z - 1] = np.dot(z_chunk, chunk_of_p) / np.sum(chunk_of_p)
    errors = []
    for i in range(n_iter):
        # setting a new z values array
        tmp_z_values = np.zeros((n_quant + 1,), dtype=np.int16)
        tmp_z_values[0] = -1
        tmp_z_values[-1] = 255
        # calculate new z values
        for z in range(1, len(z_values) - 1):
            tmp_z_values[z] = np.round((q_values[z - 1] + q_values[z]) / 2)
        if np.array_equal(tmp_z_values, z_values):
            break
        z_values = tmp_z_values

        # calculate new q values
        for z in range(1, len(z_values)):
            chunk_of_p = p_hist_image[z_values[z - 1] + 1:z_values[z] + 1]
            z_chunk = all_zs[z_values[z - 1] + 1:z_values[z] + 1]
            q_values[z - 1] = np.dot(z_chunk, chunk_of_p) / np.sum(chunk_of_p)
        # calculating the errors
        errors.append(error_calc(p_hist_image, q_values, z_values))

    lookuptable = np.zeros((256,), dtype=np.uint8)
    lookuptable[0] = np.round(q_values[0])
    for q in range(len(q_values)):
        lookuptable[z_values[q] + 1: z_values[q + 1] + 1] = np.round(
            q_values[q])
    new_image_y = (lookuptable[int_image] / 255).astype(np.float64)
    img_after_eq = new_image_y
    if im_orig.ndim == 3:
        yiq_image[:, :, 0] = new_image_y
        img_after_eq = yiq2rgb(yiq_image)
    return [img_after_eq, np.array(errors)]


def error_calc(image_histogram, q_values, z_values):
    """
    This function calculates the error between the quantized image to the original image.
    :param image_histogram:
    :param q_values:
    :param z_values:
    :return:
    """
    error_squared = 0
    nums = np.arange(256)
    for z in range(1, len(z_values)):
        new_array = np.square(nums[z_values[z - 1] + 1: z_values[z] + 1] -
                              q_values[z - 1])
        new_array2 = image_histogram[
            nums[z_values[z - 1] + 1: z_values[z] + 1]]
        error_squared += np.sum(np.dot(new_array, new_array2))
    return error_squared


def quantize_rgb(im_orig, n_quant):
    """
    This function quantizes the colors of the image, which is using less colors in the image (combining
    similar colors to one color and keeping the image as close as possible to the original). By doing so, it makes
    the image more compressible and the image's file size is reduced.
    :param im_orig:
    :param n_quant:
    :return:
    """
    initial_num_of_bins = find_the_almost_highest_numbers_of_bins(n_quant)
    reshaped_image = im_orig.reshape(-1, im_orig.shape[-1])
    if n_quant == 1:
        new_image = np.full(reshaped_image.shape,
                            np.average(reshaped_image, axis=0))
        return new_image.reshape(im_orig.shape)
    if n_quant == 3:
        bins = divide_into_3_bins(np.arange(reshaped_image.shape[0]),
                                  reshaped_image)
    else:
        bins = initial_divide(reshaped_image)
    if n_quant > 3:
        # This loop runs in initial_num_of_bins iterations which
        # is 1/8 to 1/4 of n_quant number (and after minus 1 it's non negative)
        for i in range(initial_num_of_bins - 1):
            new_bins = []
            for binn in bins:
                if binn.any():
                    new_bins.extend(divide_into_2_bins(binn, reshaped_image))
            bins = new_bins
        sorted_bins = sorted(bins, key=lambda indices: indices.size)
        num_of_bins_for_each_bin = np.full((len(sorted_bins),), 2,
                                           dtype=np.uint8)
        final_bins = []
        # This loop runs in a numner of iterations which is less than
        # n_quant number (and non negative)
        for i in range(n_quant - 2 * len(sorted_bins)):
            num_of_bins_for_each_bin[i % len(sorted_bins)] += 1
        # This loop runs in a number of iterations which is twice the
        # initial_num_of_bins which means it is 1/4 to 1/2 of n_quant number
        for j in range(len(sorted_bins)):
            if not sorted_bins[j].any():
                continue
            if num_of_bins_for_each_bin[j] == 4:
                final_bins.extend(
                    divide_into_4_bins(sorted_bins[j], reshaped_image))
            elif num_of_bins_for_each_bin[j] == 3:
                final_bins.extend(
                    divide_into_3_bins(sorted_bins[j], reshaped_image))
            else:
                final_bins.extend(
                    divide_into_2_bins(sorted_bins[j], reshaped_image))
        bins = final_bins
    # This loop runs in the same number of n_quant
    for binn in bins:
        if binn.any():
            average_color = np.average(reshaped_image[binn], axis=0)
            reshaped_image[binn] = average_color
    return reshaped_image.reshape(im_orig.shape)


def find_the_almost_highest_numbers_of_bins(n):
    """
    This function is a helper for quantize_rgb function.
    :param n:
    :return:
    """
    start = 1
    while 2 ** start < n:
        start += 1
    return start - 2


def divide_into_2_bins(bin_to_split, image):
    """
    This function is a helper for quantize_rgb function.
    :param bin_to_split:
    :param image:
    :return:
    """
    part_of_image = image[bin_to_split]
    max_range_color = find_biggest_color_range(part_of_image)
    biggest_range_color = part_of_image[:, max_range_color]
    if biggest_range_color.size != 0:
        median = np.median(biggest_range_color)
    else:
        return [np.zeros((1,)), np.zeros((1,))]
    if np.max(biggest_range_color) == np.min(biggest_range_color):
        first_half_indices = np.arange(int(biggest_range_color.size / 2))
        second_half_indices = np.arange(int(biggest_range_color.size / 2),
                                        biggest_range_color.size)
    else:
        first_half_indices = np.where(biggest_range_color <= median)[0]
        second_half_indices = np.where(biggest_range_color > median)[0]
    return [bin_to_split[first_half_indices],
            bin_to_split[second_half_indices]]


def divide_into_3_bins(bin_to_split, image):
    """
    This function is a helper for quantize_rgb function.
    :param bin_to_split:
    :param image:
    :return:
    """
    part_of_image = image[bin_to_split]
    max_range_color = find_biggest_color_range(part_of_image)
    biggest_range_color = part_of_image[:, max_range_color]
    if biggest_range_color.size != 0:
        thirds = [np.quantile(biggest_range_color, 1 / 3),
                  np.quantile(biggest_range_color, 2 / 3)]
        first_third_indices = np.where(biggest_range_color <= thirds[0])[0]
        second_third_indices = np.where((biggest_range_color > thirds[0]) & (
                    biggest_range_color < thirds[1]))[0]
        third_third_indices = np.where(biggest_range_color >= thirds[1])[0]
        return [bin_to_split[first_third_indices],
                bin_to_split[second_third_indices],
                bin_to_split[third_third_indices]]
    else:
        return [np.zeros((1,)), np.zeros((1,)), np.zeros((1,))]


def divide_into_4_bins(bin_to_split, image):
    """
    This function is a helper for quantize_rgb function.
    :param bin_to_split:
    :param image:
    :return:
    """
    first_2_bins = divide_into_2_bins(bin_to_split, image)
    return divide_into_2_bins(first_2_bins[0], image) + divide_into_2_bins(
        first_2_bins[1], image)


def find_biggest_color_range(image):
    """
    This function is a helper for quantize_rgb function.
    :param image:
    :return:
    """
    red = image[:, 0]
    green = image[:, 1]
    blue = image[:, 2]
    max_range = -1
    max_range_color = -1

    if red.size and np.max(red) - np.min(red) > max_range:
        max_range_color = 0
        max_range = np.max(red) - np.min(red)
    if green.size and np.max(green) - np.min(green) > max_range:
        max_range_color = 1
        max_range = np.max(green) - np.min(green)
    if blue.size and np.max(blue) - np.min(blue) > max_range:
        max_range_color = 2

    if max_range_color == -1:
        return 0
    return max_range_color


def initial_divide(image):
    """
    This function is a helper for quantize_rgb function.
    :param image:
    :return:
    """
    max_range_color = find_biggest_color_range(image)
    biggest_range_color = image[:, max_range_color]
    median = np.median(biggest_range_color)
    first_half_indices = np.where(biggest_range_color <= median)[0]
    second_half_indices = np.where(biggest_range_color > median)[0]
    return [first_half_indices, second_half_indices]
