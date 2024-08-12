import os, pathlib #, sys, inspect
from dataclasses import dataclass
# currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)
# sys.path.append(currentdir)
from vouchervision.general_utils import validate_dir, get_datetime

@dataclass
class Dir_Structure():
    # Home 
    run_name: str = ''
    dir_home: str = ''
    dir_project: str = ''

    # Processing dirs
    path_archival_components: str = ''
    path_config_file: str = ''

    ruler_info: str = ''
    ruler_overlay: str = ''
    ruler_processed: str = ''
    ruler_data: str = ''
    ruler_class_overlay: str = ''
    ruler_validation_summary: str = ''
    ruler_validation: str = ''

    save_per_image: str = ''
    save_per_annotation_class: str = ''
    binarize_labels: str = ''

    # logging
    path_log: str = ''
    
    def __init__(self, cfg) -> None:
        # Home 
        self.run_name = cfg['leafmachine']['project']['run_name']
        self.dir_home = cfg['leafmachine']['project']['dir_output']
        self.dir_project = os.path.join(self.dir_home,self.run_name)
        validate_dir(self.dir_home)
        self.__add_time_to_existing_project_dir()
        validate_dir(self.dir_project)

        # Processing dirs
        self.path_archival_components = os.path.join(self.dir_project,'Archival_Components')
        validate_dir(self.path_archival_components)

        self.path_config_file = os.path.join(self.dir_project,'Config_File')
        validate_dir(self.path_config_file)

        self.path_cost = os.path.join(self.dir_project,'Cost')
        validate_dir(self.path_cost)

        # Logging
        self.path_log = os.path.join(self.dir_project,'Logs')
        validate_dir(self.path_log)
        
        validate_dir(os.path.join(self.path_archival_components, 'JSON'))
        validate_dir(os.path.join(self.path_archival_components, 'labels'))

        ### Data
        self.transcription = os.path.join(self.dir_project,'Transcription') 
        validate_dir(self.transcription)
        self.transcription_ind = os.path.join(self.dir_project,'Transcription','Individual') 
        validate_dir(self.transcription_ind)

        self.transcription_ind_OCR = os.path.join(self.dir_project,'Transcription','Individual_OCR') 
        validate_dir(self.transcription_ind_OCR)
        self.transcription_ind_OCR_bounds = os.path.join(self.dir_project,'Transcription','Individual_OCR_Bounds') 
        validate_dir(self.transcription_ind_OCR_bounds)
        self.transcription_ind_OCR_helper = os.path.join(self.dir_project,'Transcription','Individual_OCR_Helper') 
        validate_dir(self.transcription_ind_OCR_helper)
        self.transcription_ind_wiki = os.path.join(self.dir_project,'Transcription','Individual_Wikipedia') 
        validate_dir(self.transcription_ind_wiki)

        self.transcription_ind_prompt = os.path.join(self.dir_project,'Transcription','Individual_Prompt') 
        validate_dir(self.transcription_ind_prompt)
        self.transcription_prompt = os.path.join(self.dir_project,'Transcription','Prompt_Template') 
        validate_dir(self.transcription_prompt)

        self.ind_redaction = os.path.join(self.dir_project, 'Redaction', 'Individual_Redacted')
        validate_dir(self.ind_redaction)

        self.save_original = os.path.join(self.dir_project,'Original_Images') 
        validate_dir(self.save_original)

        self.save_per_image = os.path.join(self.dir_project,'Cropped_Images', 'By_Image') 
        self.save_per_annotation_class = os.path.join(self.dir_project,'Cropped_Images', 'By_Class') 
        self.save_per_annotation_class = os.path.join(self.dir_project,'Cropped_Images', 'By_Class') 
        if cfg['leafmachine']['cropped_components']['save_per_image']:
            validate_dir(self.save_per_image)
        if cfg['leafmachine']['cropped_components']['save_per_annotation_class']:
            validate_dir(self.save_per_annotation_class)
        if cfg['leafmachine']['cropped_components']['binarize_labels']:
            validate_dir(self.save_per_annotation_class)

    def __add_time_to_existing_project_dir(self) -> None:
        path = pathlib.Path(self.dir_project)
        if path.exists():
            now = get_datetime()
            path = path.with_name(path.name + "_" + now)
            self.run_name = path.name
            path.mkdir()
            self.dir_project = path
        else:
            path.mkdir()
            self.dir_project = path