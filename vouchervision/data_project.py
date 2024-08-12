import os, sys, inspect, shutil, warnings
from dataclasses import dataclass, field
import pandas as pd
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from vouchervision.general_utils import import_csv, import_tsv, bcolors
from vouchervision.general_utils import Print_Verbose, print_main_warn, print_main_success, make_file_names_valid, make_images_in_dir_vertical
from vouchervision.utils_GBIF import generate_image_filename
from vouchervision.download_from_GBIF_all_images_in_file import download_all_images_from_GBIF_LM2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import fitz
from logging import Logger
from vouchervision.directory_structure_VV import Dir_Structure


def convert_pdf_to_jpg(source_pdf, destination_dir, dpi=100):
    doc = fitz.open(source_pdf)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the current page
        pix = page.get_pixmap(dpi=dpi)  # Render page to an image
        output_filename = f"{os.path.splitext(os.path.basename(source_pdf))[0]}__{10000 + page_num + 1}.jpg"
        output_filepath = os.path.join(destination_dir, output_filename)
        pix.save(output_filepath)  # Save the image
    length_doc = len(doc)  
    doc.close()
    return length_doc


@dataclass
class Project_Info():
    batch_size: int = 50
    has_valid_images: bool = True
    image_location: str = ''
    dir_images: str = ''
    path_csv_combined: str = ''
    path_csv_occ: str = ''
    path_csv_img: str = ''    
    csv_combined: str = ''
    csv_occ: str = ''
    csv_img: str = ''

    project_data: object = field(init=False)
    project_data_list: object = field(init=False)
    Dirs: object = field(init=False) 

    def __init__(self, cfg:dict, logger:Logger, dir_home:str, Dirs:Dir_Structure) -> None:
        """
        Initialize the Project_Info object

        Args:
            cfg (dict): Configuration dictionary
            logger (Logger): Python logger object
            dir_home (str): Home directory
            Dirs (object): Dir_Structure object containing project directory paths
        """
        self.Dirs = Dirs
        logger.name = 'Project Info'
        logger.info("Gathering Images and Image Metadata")
        self.logger = logger

        self.batch_size = cfg['leafmachine']['project']['batch_size']

        self.image_location = cfg['leafmachine']['project']['image_location']
        
        self.valid_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

        self.copy_images_to_project_dir(cfg['leafmachine']['project']['dir_images_local'], Dirs)

        self.make_file_names_custom(Dirs.save_original, cfg, Dirs)

        # If project is local, expect:
        #       dir with images
        #       path to images.csv
        #       path to occ.csv
        #   OR  path to combined.csv
        # if self.image_location in ['local','l','L','Local']:
        self.__import_local_files(cfg, logger, Dirs)

        # If project is GBIF, expect:
        #       Darwin Core Images (or multimedia.txt) and Occurrences file pair, either .txt or .csv
        # elif self.image_location in ['GBIF','g','G','gbif']:
        #     self.__import_GBIF_files_post_download(cfg, logger, dir_home)

        self.__make_project_dict(Dirs) #, self.batch_size)

        # Make sure image file names are legal
        make_file_names_valid(Dirs.save_original, cfg)
        
        # Make all images vertical
        make_images_in_dir_vertical(Dirs.save_original, cfg)

    @property
    def has_valid_images(self) -> bool:
        return self.check_for_images()
    
    @property
    def file_ext(self) -> str:
        return f"{['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']}"
    
    def check_for_images(self) -> bool:
        """
        Check if there are any files in the directory with valid image extensions

        Returns:
            bool: True if there are valid image files in the directory, False otherwise
        """
        for filename in os.listdir(self.dir_images):
            if filename.endswith(tuple(self.valid_extensions)):
                return True
        return False
    
    def remove_non_numbers(self, s:str) -> str:
        """
        Remove all non-numeric characters from a string

        Args:
            s (str): Input string
        
        Returns:
            str: String with only numeric characters
        """
        return ''.join([char for char in s if char.isdigit()])
    
    def copy_images_to_project_dir(self, dir_images:str, Dirs:Dir_Structure) -> None:
        """
        Copy images to the project directory

        Args:
            dir_images (str): Directory containing images
            Dirs (Dir_Structure): Directory structure object
        """

        n_total = len(os.listdir(dir_images))
        for file in tqdm(os.listdir(dir_images), desc=f'{bcolors.HEADER}     Copying images to working directory{bcolors.ENDC}', colour="white", position=0, total=n_total):
            source = os.path.join(dir_images, file)
            # Check if file is a PDF
            if file.lower().endswith('.pdf'):
                # Convert PDF pages to JPG images
                n_pages = convert_pdf_to_jpg(source, Dirs.save_original)
                self.logger.info(f"Converted {n_pages} pages to JPG from PDF: {file}")
            else:
                # Copy non-PDF files directly
                destination = os.path.join(Dirs.save_original, file)
                shutil.copy(source, destination)
        
    def make_file_names_custom(self, dir_images:str, cfg:dict, Dirs:Dir_Structure) -> None: 
        """
        Create custom file names for images based on configuration settings
        May remove prefixes, suffixes, and/or non-numeric characters from file names

        Args:
            dir_images (str): Directory containing images
            cfg (dict): Configuration dictionary
            Dirs (Dir_Structure): Directory structure object
        """
        n_total = len(os.listdir(dir_images))
        for file in tqdm(os.listdir(dir_images), desc=f'{bcolors.HEADER}     Creating Catalog Number from file name{bcolors.ENDC}',colour="green",position=0,total = n_total):
            if cfg['leafmachine']['project']['catalog_numerical_only'] or cfg['leafmachine']['project']['prefix_removal'] or cfg['leafmachine']['project']['suffix_removal']:
                name = Path(file).stem
                ext = Path(file).suffix
                if cfg['leafmachine']['project']['prefix_removal']:
                    name_cleaned = name.replace(cfg['leafmachine']['project']['prefix_removal'], "")
                if cfg['leafmachine']['project']['suffix_removal']:
                    name_cleaned = name.replace(cfg['leafmachine']['project']['suffix_removal'], "")
                if cfg['leafmachine']['project']['catalog_numerical_only']:
                    name_cleaned = self.remove_non_numbers(name)
                name_new = ''.join([name_cleaned,ext])
                i = 0
                try:
                    os.rename(os.path.join(dir_images,file), os.path.join(dir_images,name_new))
                except:
                    warnings.warn("WARNING: duplicate file names will result given the current selections for 'prefix_removal', 'suffix_removal', or 'catalog_numerical_only'. Change them before continuing.")
                    warnings.warn("The affected file name has not been changed.")

    def __create_combined_csv(self):
        """
        Create a combined CSV file from the occurrence and image metadata files

        Returns:
            str: Path to the combined CSV file
        """
        self.csv_img = self.csv_img.rename(columns={"gbifID": "gbifID_images"}) 
        self.csv_img = self.csv_img.rename(columns={"identifier": "url"}) 

        combined = pd.merge(self.csv_img, self.csv_occ, left_on='gbifID_images', right_on='gbifID')
        names_list = combined.apply(generate_image_filename, axis=1, result_type='expand')

        # Select columns 7, 0, 1
        selected_columns = names_list.iloc[:,[7,0,1]]
        # Rename columns
        selected_columns.columns = ['fullname','filename_image','filename_image_jpg']

        self.csv_combined = pd.concat([selected_columns, combined], axis=1)

        new_name = ''.join(['combined_', os.path.basename(self.path_csv_occ).split('.')[0], '_', os.path.basename(self.path_csv_img).split('.')[0], '.csv'])
        self.path_csv_combined = os.path.join(os.path.dirname(self.path_csv_occ), new_name)
        self.csv_combined.to_csv(self.path_csv_combined, mode='w', header=True, index=False)
        return self.path_csv_combined

    def __import_local_files(self, cfg, logger, Dirs):
        # Images
        if cfg['leafmachine']['project']['dir_images_local'] is None:
            self.dir_images = None
        else:
            self.dir_images = Dirs.save_original 
        
        # CSV import
        # Combined
        try:
            if cfg['leafmachine']['project']['path_combined_csv_local'] is None:
                self.csv_combined = None
                self.path_csv_combined = None
            else:
                self.path_csv_combined = cfg['leafmachine']['project']['path_combined_csv_local']
                self.csv_combined = import_csv(self.path_csv_combined)
            # Occurrence
            if cfg['leafmachine']['project']['path_occurrence_csv_local'] is None:
                self.csv_occ = None
                self.path_csv_occ = None
            else:
                self.path_csv_occ = cfg['leafmachine']['project']['path_occurrence_csv_local']
                self.csv_occ = import_csv(self.path_csv_occ)
            # Images/metadata
            if cfg['leafmachine']['project']['path_images_csv_local'] is None:
                self.path_csv_img = None
                self.path_csv_img = None
            else:
                self.path_csv_img = cfg['leafmachine']['project']['path_images_csv_local']
                self.csv_img = import_csv(self.path_csv_img)

            # Create combined if it's missing
            if self.csv_combined is None:
                if cfg['leafmachine']['project']['path_combined_csv_local'] is not None:
                    # Print_Verbose(cfg, 2, 'Combined CSV file not provided, creating it now...').bold()
                    logger.info('Combined CSV file not provided, creating it now...')
                    location = self.__create_combined_csv()
                    # Print_Verbose(cfg, 2, ''.join(['Combined CSV --> ',location])).green()
                    logger.info(''.join(['Combined CSV --> ',location]))

                else:
                    # Print_Verbose(cfg, 2, 'Combined CSV file not available or provided. Skipped record import.').bold()
                    logger.info('Combined CSV file not available or provided. Skipped record import.')
            else:
                # Print_Verbose(cfg, 2, ''.join(['Combined CSV --> ',self.path_csv_combined])).green()
                logger.info(''.join(['Combined CSV --> ',self.path_csv_combined]))
        except:
            pass

        # Print_Verbose(cfg, 2, ''.join(['Image Directory --> ',self.dir_images])).green()
        logger.info(''.join(['Image Directory --> ',Dirs.save_original]))

    def process_in_batches(self, cfg):
        batch_size = cfg['leafmachine']['project']['batch_size']
        self.project_data_list = []
        keys = list(self.project_data.keys())
        num_batches = len(keys) // batch_size + 1
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_keys = keys[start:end]
            batch = {key: self.project_data[key] for key in batch_keys}
            self.project_data_list.append(batch)
        return num_batches, len(self.project_data)

    def __make_project_dict(self, Dirs):
        self.project_data = {}
        invalid_dir = None

        for img in os.listdir(Dirs.save_original):
            img_split, ext = os.path.splitext(img)
            if ext in self.valid_extensions:
                with Image.open(os.path.join(Dirs.save_original, img)) as im:
                    _, ext = os.path.splitext(img)
                    if ext not in ['.jpg']:
                        im = im.convert('RGB')
                        new_img_name = ''.join([img_split, '.jpg'])
                        im.save(os.path.join(Dirs.save_original, new_img_name), quality=100)
                        self.project_data[img_split] = {}

                        # move the original file to the INVALID_FILE directory
                        if invalid_dir is None:
                            invalid_dir = os.path.join(os.path.dirname(Dirs.save_original), 'INVALID_FILES')
                            os.makedirs(invalid_dir, exist_ok=True)

                        # skip if the file already exists in the INVALID_FILE directory
                        if not os.path.exists(os.path.join(invalid_dir, img)):
                            shutil.move(os.path.join(Dirs.save_original, img), os.path.join(invalid_dir, img))

                        img = new_img_name
                img_name = os.path.splitext(img)[0]
                self.project_data[img_split] = {}
            else:
                # if the file has an invalid extension, move it to the INVALID_FILE directory
                if invalid_dir is None:
                    invalid_dir = os.path.join(os.path.dirname(Dirs.save_original), 'INVALID_FILES')
                    os.makedirs(invalid_dir, exist_ok=True)

                # skip if the file already exists in the INVALID_FILE directory
                if not os.path.exists(os.path.join(invalid_dir, img)):
                    shutil.move(os.path.join(Dirs.save_original, img), os.path.join(invalid_dir, img))

    def add_records_to_project_dict(self):
        for img in os.listdir(self.Dirs.save_original):
            if (img.endswith(".jpg") or img.endswith(".jpeg")):
                img_name = str(img.split('.')[0])
                try:
                    self.project_data[img_name]['GBIF_Record'] = self.__get_data_from_combined(img_name)
                except:
                    self.project_data[img_name]['GBIF_Record'] = None

    def __get_data_from_combined(self, img_name):
        df = pd.DataFrame(self.csv_combined)
        row = df[df['filename_image'] == img_name].head(1).to_dict()
        return row


class Project_Stats():
    specimens = 0
    
    rulers = 0
    

    def __init__(self, cfg, logger, dir_home) -> None:
        logger.name = 'Project Info'
        logger.info("Gathering Images and Image Metadata")