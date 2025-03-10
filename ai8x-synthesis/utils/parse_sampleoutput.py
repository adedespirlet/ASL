###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
##################################################################################################

"""
Utility functions for parsing izer generated sampleoutput.h
"""


def twos_compl_to_int(val, nbits, base):
    """
    Returns signed integer value for a given value:

    :param val Value in two signed complement form, in string type
    :param nbits the number of bits val represents
    :param base base of val
    """
    i = int(val, base)
    if i >= 2 ** (nbits - 1):
        i -= 2 ** nbits
    return i


def parse_sampleoutput_to_dictionary(sampleoutput_h_file_path):
    """
    Parsed auto generated sampleoutput.h file (genearted by izer).
    Returns a dictionary with keys of address values of emmory blocks and
    values as the list of read strings in the file for given address.
    """

    out_dict = {}

    with open(sampleoutput_h_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    data_sample_out = []
    for line in lines[1:-1]:
        values = line.split(',')[:-1]
        for val in values:
            data_sample_out.append(val.strip())

    addr = None
    d_len = None
    mask = None

    for val in data_sample_out:
        if addr is None:
            addr = val
        elif mask is None:
            mask = val

        elif d_len is None:
            d_len = int(val, 0)
            d_to_read = d_len
            address_contents = []

        else:
            address_contents.append(val)
            d_to_read -= 1
            if d_to_read == 0:
                out_dict[addr] = address_contents

                addr = None
                d_len = None
                mask = None

    return out_dict


def process_four_channel_data_from_word(word_value):
    """
    For 8 bit outputs, izer combines 4 of them and places all into a single memory word.
    This function seperates out the 4 channels from a given word value.
    word_value must be a hexadecimal string.
    """

    ch0 = int(word_value, 16) & 0x000000ff
    ch0 = f'{ch0:>08b}'
    ch1 = (int(word_value, 16) & 0x0000ff00) >> 8
    ch1 = f'{ch1:>08b}'
    ch2 = (int(word_value, 16) & 0x00ff0000) >> 16
    ch2 = f'{ch2:>08b}'
    ch3 = (int(word_value, 16) & 0xff000000) >> 24
    ch3 = f'{ch3:>08b}'

    # From Q7 8 bits to float:
    ch0 = twos_compl_to_int(ch0, 8, 2)
    ch1 = twos_compl_to_int(ch1, 8, 2)
    ch2 = twos_compl_to_int(ch2, 8, 2)
    ch3 = twos_compl_to_int(ch3, 8, 2)

    return [ch0, ch1, ch2, ch3]
