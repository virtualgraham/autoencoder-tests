import subprocess
from os import listdir
from os.path import isfile, join
import cv2

drives = [
    ('road', '2011_09_26', '0015'), 
    ('road', '2011_09_26', '0027'),
    ('road', '2011_09_26', '0028'),
    ('road', '2011_09_26', '0029'),
    ('road', '2011_09_26', '0032'),
    ('road', '2011_09_26', '0052'),
    ('road', '2011_09_26', '0070'),
    ('road', '2011_09_26', '0101'),
    ('road', '2011_09_29', '0004'),
    ('road', '2011_09_30', '0016'),
    ('road', '2011_10_03', '0042'),
    ('residential', '2011_09_26', '0019'), 
    ('residential', '2011_09_26', '0020'), 
    ('residential', '2011_09_26', '0022'), 
    ('residential', '2011_09_26', '0023'), 
    ('residential', '2011_09_26', '0035'), 
    ('residential', '2011_09_26', '0036'), 
    ('residential', '2011_09_26', '0039'), 
    ('residential', '2011_09_26', '0046'), 
    ('residential', '2011_09_26', '0061'), 
    ('residential', '2011_09_26', '0079'), 
    ('residential', '2011_09_26', '0086'), 
    ('residential', '2011_09_26', '0087'), 
    ('residential', '2011_09_30', '0018'), 
    ('residential', '2011_09_30', '0020'), 
    ('residential', '2011_09_30', '0027'), 
    ('residential', '2011_09_30', '0028'), 
    ('residential', '2011_09_30', '0033'), 
    ('residential', '2011_09_30', '0034'), 
    ('residential', '2011_10_03', '0027'), 
    ('residential', '2011_10_03', '0034'), 
    ('city', '2011_09_26', '0001'), 
    ('city', '2011_09_26', '0002'), 
    ('city', '2011_09_26', '0005'), 
    ('city', '2011_09_26', '0009'), 
    ('city', '2011_09_26', '0011'), 
    ('city', '2011_09_26', '0013'), 
    ('city', '2011_09_26', '0014'), 
    ('city', '2011_09_26', '0017'), 
    ('city', '2011_09_26', '0018'), 
    ('city', '2011_09_26', '0048'), 
    ('city', '2011_09_26', '0051'), 
    ('city', '2011_09_26', '0056'), 
    ('city', '2011_09_26', '0057'), 
    ('city', '2011_09_26', '0059'), 
    ('city', '2011_09_26', '0084'), 
    ('city', '2011_09_26', '0091'), 
    ('city', '2011_09_26', '0093'), 
    ('city', '2011_09_26', '0095'), 
    ('city', '2011_09_26', '0096'),
    ('city', '2011_09_26', '0104'),
    ('city', '2011_09_26', '0106'),
    ('city', '2011_09_26', '0113'),
    ('city', '2011_09_26', '0117'),
    ('city', '2011_09_28', '0001'),
    ('city', '2011_09_28', '0002'),
    ('city', '2011_09_29', '0026'),
    ('city', '2011_09_29', '0071'),
    ('campus', '2011_09_28', '0047'), 
    ('campus', '2011_09_28', '0045'), 
    ('campus', '2011_09_28', '0043'), 
    ('campus', '2011_09_28', '0039'), 
    ('campus', '2011_09_28', '0038'), 
    ('campus', '2011_09_28', '0037'), 
    ('campus', '2011_09_28', '0035'), 
    ('campus', '2011_09_28', '0034'), 
    ('campus', '2011_09_28', '0021'), 
    ('campus', '2011_09_28', '0016'),
]

test_drives = [
    ('road', '2011_09_29', '0004'),
    ('road', '2011_09_26', '0028'),
    ('residential', '2011_09_26', '0035'), 
    ('residential', '2011_09_26', '0046'),
    ('residential', '2011_09_26', '0079'), 
    ('residential', '2011_09_30', '0034'), 
    ('residential', '2011_10_03', '0034'), 
    ('city', '2011_09_26', '0009'),
    ('city', '2011_09_26', '0095'), 
    ('city', '2011_09_26', '0057'), 
    ('city', '2011_09_29', '0071'),
    ('city', '2011_09_26', '0018'), 
    ('city', '2011_09_28', '0002'),
    ('campus', '2011_09_28', '0021'), 
    ('campus', '2011_09_28', '0038'),
    
]

train_drives = [
    ('road', '2011_09_26', '0015'), 
    ('road', '2011_09_26', '0027'),
    ('road', '2011_09_26', '0029'),
    ('road', '2011_09_26', '0032'),
    ('road', '2011_09_26', '0052'),
    ('road', '2011_09_26', '0070'),
    ('road', '2011_09_26', '0101'),
    ('road', '2011_09_30', '0016'),
    ('road', '2011_10_03', '0042'),
    ('residential', '2011_09_26', '0019'), 
    ('residential', '2011_09_26', '0020'), 
    ('residential', '2011_09_26', '0022'), 
    ('residential', '2011_09_26', '0023'), 
    ('residential', '2011_09_26', '0036'), 
    ('residential', '2011_09_26', '0039'), 
    ('residential', '2011_09_26', '0061'), 
    ('residential', '2011_09_26', '0086'), 
    ('residential', '2011_09_26', '0087'), 
    ('residential', '2011_09_30', '0018'), 
    ('residential', '2011_09_30', '0020'), 
    ('residential', '2011_09_30', '0027'), 
    ('residential', '2011_09_30', '0028'), 
    ('residential', '2011_09_30', '0033'), 
    ('residential', '2011_10_03', '0027'), 
    ('city', '2011_09_26', '0001'), 
    ('city', '2011_09_26', '0002'), 
    ('city', '2011_09_26', '0005'), 
    ('city', '2011_09_26', '0011'), 
    ('city', '2011_09_26', '0013'), 
    ('city', '2011_09_26', '0014'), 
    ('city', '2011_09_26', '0017'), 
    ('city', '2011_09_26', '0048'), 
    ('city', '2011_09_26', '0051'), 
    ('city', '2011_09_26', '0056'), 
    ('city', '2011_09_26', '0059'), 
    ('city', '2011_09_26', '0084'), 
    ('city', '2011_09_26', '0091'), 
    ('city', '2011_09_26', '0093'), 
    ('city', '2011_09_26', '0096'),
    ('city', '2011_09_26', '0104'),
    ('city', '2011_09_26', '0106'),
    ('city', '2011_09_26', '0113'),
    ('city', '2011_09_26', '0117'),
    ('city', '2011_09_28', '0001'),
    ('city', '2011_09_29', '0026'),
    ('campus', '2011_09_28', '0047'), 
    ('campus', '2011_09_28', '0045'), 
    ('campus', '2011_09_28', '0043'), 
    ('campus', '2011_09_28', '0039'), 
    ('campus', '2011_09_28', '0037'), 
    ('campus', '2011_09_28', '0035'), 
    ('campus', '2011_09_28', '0034'), 
    ('campus', '2011_09_28', '0016'),
]

kitti_dir = '../datasets/kitti/'
zips_dir = kitti_dir + 'zips/'
format_dir = kitti_dir + 'format1/'
encoded_dir = kitti_dir + 'encoded_v1/'

def list_drive_encoded_files(drive):

    paths = list()

    drive_date = drive[1]
    drive_number = drive[2]

    drive_dir = encoded_dir + drive_date + '_' + drive_number + '/'

    drive_encoded_files = [f for f in listdir(drive_dir) if isfile(join(drive_dir, f))]

    for filename in drive_encoded_files:

        source_path = join(drive_dir, filename)
        paths.append(source_path)

    paths.sort()

    return paths


def get_encoded_sequences_from_drive(drive, seq_len):

    encoded_files = list_drive_encoded_files(drive)

    seq_count = len(encoded_files) - seq_len + 1

    sequences = list()

    for i in range(0, seq_count):

        seq = list()

        for j in range(0, seq_len):

            seq.append(encoded_files[i+j])

        sequences.append(seq)

    return sequences


def get_encoded_sequences_from_drives(drives_array, seq_len):

    sequences = list()

    for drive in drives_array:

        drive_sequences = get_encoded_sequences_from_drive(drive, seq_len)
        sequences.extend(drive_sequences)

    return sequences

# returns a list containing for each drive a list containing for each sequence a list containing npy filenames
def list_training_sequences(seq_len):

    return get_encoded_sequences_from_drives(train_drives, seq_len)


def list_testing_sequences(seq_len):

    return get_encoded_sequences_from_drives(test_drives, seq_len)

def list_all_sequences(seq_len):

    return get_encoded_sequences_from_drives(drives, seq_len)


def list_drive_png_files(drive):

    paths = list()

    drive_date = drive[1]
    drive_number = drive[2]

    drive_dir = format_dir + drive_date + '_' + drive_number + '/'

    drive_pngs = [f for f in listdir(drive_dir) if isfile(join(drive_dir, f))]

    for filename in drive_pngs:

        source_path = join(drive_dir, filename)
        paths.append(source_path)

    paths.sort()

    return paths


def list_png_files(drives_array):
    
    paths = list()

    for drive in drives_array:

        drive_paths = list_drive_png_files(drive)
        paths.extend(drive_paths)

    return paths


def list_training_files():

    return list_png_files(train_drives)


def list_testing_files():

    return list_png_files(test_drives)

def list_all_files():

    return list_png_files(drives)




    

def convert():

    for drive in drives:

        drive_category = drive[0]
        drive_date = drive[1]
        drive_number = drive[2]

        source_dir = '../datasets/kitti/seqs/' + drive_date + '_' + drive_number + '/left/'
        dest_dir = '../datasets/kitti/format1/' + drive_date + '_' + drive_number + '/'

        print subprocess.check_output(['mkdir', '--parents', dest_dir])

        source_pngs = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]

        for filename in source_pngs:

            source_path = join(source_dir, filename)
            print(source_path)

            # open png from src
            orig_image = cv2.imread(source_path)

            # resize
            resize_image = cv2.resize(orig_image,(423,128))
            
            # crop
            crop_image = resize_image[:,20:404]

            # save to dest dir
            cv2.imwrite(dest_dir + filename, crop_image)


def download_raw_data():

    for drive in drives:

        drive_category = drive[0]
        drive_date = drive[1]
        drive_number = drive[2]

        zip_file_source = 'gs://numnum/kitti/' + drive_category + '/' + drive_date + '_drive_' + drive_number + '_sync.zip'
        zip_file_dest = zips_dir + drive_date + '_drive_' + drive_number + '_sync.zip'
        unziped_file_dir = zips_dir + drive_date

        print('zip_file_source', zip_file_source)
        print('zip_file_dest', zip_file_dest)
        print('unziped_file_dir', unziped_file_dir)

        print subprocess.check_output(['gsutil','cp', zip_file_source, zip_file_dest])
        print subprocess.check_output(['unzip', zip_file_dest, '-d', zips_dir])

        right_color_img_dir = zips_dir + drive_date + '/' + drive_date + '_drive_' + drive_number + '_sync/image_02/data/'
        left_color_img_dir = zips_dir + drive_date + '/' + drive_date + '_drive_' + drive_number + '_sync/image_03/data/'

        print('left_color_img_dir', left_color_img_dir)

        pngs = [f for f in listdir(left_color_img_dir) if isfile(join(left_color_img_dir, f))]

        for filename in pngs:
            source_path = join(left_color_img_dir, filename)
            dest_dir = '../datasets/kitti/seqs/' + drive_date + '_' + drive_number + '/left'
            dest_path = join(dest_dir, filename)

            print('dest_dir', dest_dir)
            print('dest_path', dest_path)

            print subprocess.check_output(['mkdir', '--parents', dest_dir])
            print subprocess.check_output(['mv', source_path, dest_path])

        print subprocess.check_output(['rm', '-r', unziped_file_dir])
        print subprocess.check_output(['rm', zip_file_dest,])