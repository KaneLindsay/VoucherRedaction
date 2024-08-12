import os, io, statistics, json, cv2
import regex as re
import colorsys
import torch
import warnings
from statistics import mean 
from google.cloud import vision
from google.cloud import vision_v1p3beta1 as vision_beta
from google.cloud.vision_v1p3beta1.types import Block, Paragraph, Word, Symbol, Vertex, Page
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from OCR_Florence_2 import FlorenceOCR
from OCR_GPT4oMini import GPT4oMiniOCR
from logging import Logger
# from vouchervision.VoucherVision_GUI import JSONReport # CIRCULAR IMPORT
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List

### LLaVA should only be installed if the user will actually use it.
### It requires the most recent pytorch/Python and can mess with older systems


"""
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
"""

class OCREngine:
    """
    OCR Engine class for extracting text from images with various OCR models available.

    Current available methods:
    - Google Cloud Vision API
    - CRAFT (Character Region Awareness for Text Detection)
    - trOCR (Transformer-based Optical Character Recognition)
    - LLaVA (Language and Vision for All)
    - Florence-2 (Microsoft's OCR model)
    - GPT-4o-mini (OpenAI's OCR model)
    """

    VALID_GRANULARITIES = ["all", "page", "block", "paragraph", "line", "word", "character"]

    class Line():
        """
        Class representing a line of text in a document

        Attributes:
            words (List): A list of Word objects
            bounding_box (vision_beta.BoundingPoly): The bounding box of the line
            confidence (float): The confidence score of the line

        Methods:
            add_word: Add a Word object to the line
            update_confidence: Update the confidence score of the line
            set_bounding_box: Set the bounding box of the line
            update_bounding_box: Update the bounding box of the line
        """

        def __init__(self, words:List[Word]=[]) -> None:
            self.words = words
            self.bounding_box = None
            self.confidence = 0.0

            self._update_bounding_box()
            self._update_confidence()

        def __repr__(self) -> str:
            return f"Line(words={self.words}, bounding_box={self.bounding_box}, confidence={self.confidence})"

        def  __str__(self) -> str:
            return OCREngine.get_element_text(self.words)

        def add_word(self, word:Word) -> None:
            """
            Add a google.cloud.vision_v1.types.Word object to the line

            Args:
                word (google.cloud.vision_v1.types.Word): The Word object to add
            """
            self.words.append(word)
            self._update_confidence()
            self._update_bounding_box()

        def set_bounding_box(self, vertices:List[Vertex]) -> None:
            """
            Args:
                vertices (List): A list of vision_beta.Vertex objects representing the bounding box corners in order [top-left, top-right, bottom-right, bottom-left]
            """
            self.bounding_box = vision_beta.BoundingPoly(vertices=vertices)

        def _update_confidence(self) -> None:
            """
            Update the confidence score of the line based on the average confidence score of the words
            """
            if len(self.words) == 0:
                return
            else:
                self.confidence = sum([word.confidence for word in self.words])/len(self.words)

        def _update_bounding_box(self) -> None:
            """
            Update the bounding box of the line by merging the bounding boxes of the words
            """
            if len(self.words) == 0:
                return
            else:
                # Merge bounding boxes of words by finding the min/max x/y coordinates
                min_x = min([word.bounding_box.vertices[0].x for word in self.words])
                min_y = min([word.bounding_box.vertices[0].y for word in self.words])
                max_x = max([word.bounding_box.vertices[2].x for word in self.words])
                max_y = max([word.bounding_box.vertices[2].y for word in self.words])
                self.bounding_box = vision_beta.BoundingPoly(vertices=[vision_beta.Vertex(x=min_x, y=min_y),
                                                                    vision_beta.Vertex(x=max_x, y=min_y),
                                                                    vision_beta.Vertex(x=max_x, y=max_y),
                                                                    vision_beta.Vertex(x=min_x, y=max_y)])

    def __init__(self, logger:Logger, json_report:object, dir_home:str, is_hf:bool, path:str, cfg:dict, trOCR_model_version:str, trOCR_model:str, trOCR_processor, device:str, redactor:object=None, redaction_granularity:str='all') -> None:
        """
        Initialize the OCR engine

        Args:
            logger (Logger): Logger object
            json_report (JSONReport): JSONReport object
            dir_home (str): Path to the home directory
            is_hf (bool): Whether the OCR engine is running on Hugging Face
            path (str): Path to the image file
            cfg (dict): Configuration dictionary
            trOCR_model_version (str): Version of the trOCR model
            trOCR_model (): 
            trOCR_processor (): Processor object for the trOCR model
            device (str): Device to run the trOCR model on
        """
        self.is_hf = is_hf
        self.logger = logger

        self.json_report = json_report

        self.path = path
        self.cfg = cfg
        self.do_use_trOCR = self.cfg['leafmachine']['project']['do_use_trOCR']
        self.do_use_florence = self.cfg['leafmachine']['project']['do_use_florence']
        self.OCR_option = self.cfg['leafmachine']['project']['OCR_option']
        self.double_OCR = self.cfg['leafmachine']['project']['double_OCR']
        self.dir_home = dir_home

        # Initialize TrOCR components
        self.trOCR_model_version = trOCR_model_version
        self.trOCR_processor = trOCR_processor
        self.trOCR_model = trOCR_model
        self.device = device

        self.OCR_JSON_to_file = {}

        # for paid vLM OCR like GPT-vision
        self.cost = 0.0
        self.tokens_in = 0
        self.tokens_out = 0

        # Redactor object for censoring and classifying text
        self.redactor = redactor
        self.redaction_granularity = redaction_granularity
        self.image_redacted = None

        # Attributes for storing handwritten OCR model results
        self.hand_cleaned_text = None
        self.hand_organized_text = None
        self.hand_text_to_box_mapping = None

        # Attributes for storing standard OCR model results
        self.normal_cleaned_text = None
        self.normal_organized_text = None
        self.normal_text_to_box_mapping = None

        # Attributes for storing TrOCR model results
        self.trOCR_texts = None
        self.trOCR_text_to_box_mapping = None
        self.trOCR_bounds_flat = None
        self.trOCR_height = None
        self.trOCR_confidences = None
        self.trOCR_characters = None

        # Initialize OCR model options
        self.set_client()
        self.init_florence()
        self.init_gpt_4o_mini()
        self.init_craft()

        self.multimodal_prompt = """I need you to transcribe all of the text in this image. 
        Place the transcribed text into a JSON dictionary with this form {"Transcription_Printed_Text": "text","Transcription_Handwritten_Text": "text"}"""
        self.init_llava()

    """ Static Methods """
    @staticmethod
    def get_element_text(page_element:Page|Block|Paragraph|Word|Symbol) -> str:
        """
        Accepts a page element and returns the text string from the element by recursively traversing the element's children.

        Heirarchy of page elements in Google OCR API:
        Page -> Block -> Paragraph -> Word -> Symbol

        Args:
            page_element : The page element to extract text from
        
        Returns:
            (str): The collected text from the page element
        """

        if hasattr(page_element, "blocks"):
            return "\n\n".join([OCREngine.get_element_text(block) for block in page_element.blocks])
        elif hasattr(page_element, "paragraphs"):
            return "\n".join([OCREngine.get_element_text(paragraph) for paragraph in page_element.paragraphs])
        elif hasattr(page_element, "words"):
            return " ".join([OCREngine.get_element_text(word) for word in page_element.words])
        elif hasattr(page_element, "symbols"):
            return "".join([symbol.text for symbol in page_element.symbols])
        elif hasattr(page_element, "text"):
            return page_element.text
        else:
            return ""

    @staticmethod
    def get_element_vertices(page_element:Page|Block|Paragraph|Word|Symbol) -> List[dict]:
        return [{"x": vertex.x, "y": vertex.y} for vertex in page_element.bounding_box.vertices]
    
    @staticmethod
    def has_line_break(symbol:Symbol) -> bool:
        """
        Check if a symbol is a line break

        Args:
            symbol (google.cloud.vision_v1.types.Symbol): The symbol to check
        
        Returns:
            (bool): True if the symbol is a line break, else False
        """
        break_types = [vision_beta.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK,
                        vision_beta.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                        vision_beta.TextAnnotation.DetectedBreak.BreakType.HYPHEN]
        return symbol.property.detected_break.type in break_types

    @staticmethod
    def paragraph_to_lines(paragraph:Paragraph) -> List[Line]:
        """
        Convert a paragraph to a list of lines by splitting on line breaks

        Args:
            paragraph: The paragraph to split into lines
        
        Returns:
            (List): A list of lines
        """
        lines = []
        words = []
        for word in paragraph.words:
            words.append(word)
            for symbol in word.symbols:
                if OCREngine.has_line_break(symbol):
                    lines.append(OCREngine.Line(words=words))
                    words = []
                    break
        return lines

    @staticmethod
    def confidence_to_color(confidence):
        hue = (confidence - 0.5) * 120 / 0.5
        r, g, b = colorsys.hls_to_rgb(hue/360, 0.5, 1)
        return (int(r*255), int(g*255), int(b*255))

    """ Instance Methods """

    def set_client(self) -> None:
        """
        Set the client for the OCR engine using Google Cloud Vision API credentials
        """
        # TODO: These cases are the same, whats the point?
        if self.is_hf:
            self.client_beta = vision_beta.ImageAnnotatorClient(credentials=self.get_google_credentials())
            self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())
        else:
            self.client_beta = vision_beta.ImageAnnotatorClient(credentials=self.get_google_credentials()) 
            self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())

    def get_google_credentials(self) -> Credentials:
        """
        Retrieve the Google Cloud Vision API credentials from the environment variable 'GOOGLE_APPLICATION_CREDENTIALS'

        Returns (google.oauth2.credentials.Credentials): Google Cloud Vision API credentials
        """
        creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
        return credentials
    
    def init_craft(self) -> None:
        """
        Initialize the CRAFT OCR engine with craftnet and refinenet models
        """
        if 'CRAFT' in self.OCR_option:
            from craft_text_detector import load_craftnet_model, load_refinenet_model

            try:
                self.refine_net = load_refinenet_model(cuda=True)
                self.use_cuda = True
            except:
                self.refine_net = load_refinenet_model(cuda=False)
                self.use_cuda = False

            if self.use_cuda:
                self.craft_net = load_craftnet_model(weight_path=os.path.join(self.dir_home,'vouchervision','craft','craft_mlt_25k.pth'), cuda=True)
            else:
                self.craft_net = load_craftnet_model(weight_path=os.path.join(self.dir_home,'vouchervision','craft','craft_mlt_25k.pth'), cuda=False)

    def init_florence(self):
        """
        Initialize Microsoft's Florence model for the OCR with Region task
        """
        if 'Florence-2' in self.OCR_option:
            self.Florence = FlorenceOCR(logger=self.logger, model_id=self.cfg['leafmachine']['project']['florence_model_path'])

    def init_gpt_4o_mini(self):
        """
        Initialise OpenAI's GPT-4o-mini model
        """
        if 'GPT-4o-mini' in self.OCR_option:
            self.GPTmini = GPT4oMiniOCR(api_key = os.getenv('OPENAI_API_KEY'))

    def init_llava(self):
        """
        Initialize VicunaAI's LLaVA model for OCR
        """
        if 'LLaVA' in self.OCR_option:
            from vouchervision.OCR_llava import OCRllava

            self.model_path = "liuhaotian/" + self.cfg['leafmachine']['project']['OCR_option_llava']
            self.model_quant = self.cfg['leafmachine']['project']['OCR_option_llava_bit']
            
            if self.json_report:
                self.json_report.set_text(text_main=f'Loading LLaVA model: {self.model_path} Quantization: {self.model_quant}')

            if self.model_quant == '4bit':
                use_4bit = True
            elif self.model_quant == 'full':
                use_4bit = False
            else:
                self.logger.info(f"Provided model quantization invlid. Using 4bit.")
                use_4bit = True

            self.Llava = OCRllava(self.logger, model_path=self.model_path, load_in_4bit=use_4bit, load_in_8bit=False)

    def init_gemini_vision(self):
        # TODO: Implement Gemini Vision model initialization
        pass

    def init_gpt4_vision(self):
        # TODO: Implement GPT-4 Vision model initialization
        pass
            
    def detect_text_craft(self):
        """
        Detect text in the image using the CRAFT OCR engine
        Uses placeholders for individual character detection and confidence as CRAFT does not provide this information
        """
        from craft_text_detector import read_image, get_prediction

        # Perform prediction using CRAFT
        image = read_image(self.path)

        link_threshold = 0.85
        text_threshold = 0.4
        low_text = 0.4

        if self.use_cuda:
            self.prediction_result = get_prediction(
                image=image,
                craft_net=self.craft_net,
                refine_net=self.refine_net,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                cuda=True,
                long_size=1280
            )
        else:
            self.prediction_result = get_prediction(
                image=image,
                craft_net=self.craft_net,
                refine_net=self.refine_net,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                cuda=False,
                long_size=1280
            )

        # Initialize metadata structures
        bounds = []
        bounds_word = []  # CRAFT gives bounds for text regions, not individual words
        text_to_box_mapping = []
        bounds_flat = []
        height_flat = []
        confidences = []  # CRAFT does not provide confidences per character, so this might be uniformly set or estimated
        characters = []  # Simulating as CRAFT doesn't provide character-level details
        organized_text = ""
        
        total_b = len(self.prediction_result["boxes"])
        i=0
        # Process each detected text region
        for box in self.prediction_result["boxes"]:
            i+=1
            if self.json_report:
                self.json_report.set_text(text_main=f'Locating text using CRAFT --- {i}/{total_b}')

            vertices = [{"x": int(vertex[0]), "y": int(vertex[1])} for vertex in box]
            
            # Simulate a mapping for the whole detected region as a word
            text_to_box_mapping.append({
                "vertices": vertices,
                "text": "detected_text"  # Placeholder, as CRAFT does not provide the text content directly
            })

            # Assuming each box is a word for the sake of this example
            bounds_word.append({"vertices": vertices})

            # For simplicity, we're not dividing text regions into characters as CRAFT doesn't provide this
            # Instead, we create a single large 'character' per detected region
            bounds.append({"vertices": vertices})
            
            # Simulate flat bounds and height for each detected region
            x_positions = [vertex["x"] for vertex in vertices]
            y_positions = [vertex["y"] for vertex in vertices]
            min_x, max_x = min(x_positions), max(x_positions)
            min_y, max_y = min(y_positions), max(y_positions)
            avg_height = max_y - min_y
            height_flat.append(avg_height)
            
            # Assuming uniform confidence for all detected regions
            confidences.append(1.0)  # Placeholder confidence
            
            # Adding dummy character for each box
            characters.append("X")  # Placeholder character
            
            # Organize text as a single string (assuming each box is a word)
            # organized_text += "detected_text "  # Placeholder text    

        # Update class attributes with processed data
        self.normal_bounds = bounds
        self.normal_bounds_word = bounds_word
        self.normal_text_to_box_mapping = text_to_box_mapping
        self.normal_bounds_flat = bounds_flat  # This would be similar to bounds if not processing characters individually
        self.normal_height = height_flat
        self.normal_confidences = confidences
        self.normal_characters = characters
        self.normal_organized_text = organized_text.strip() 
    
    def detect_text_with_trOCR_using_google_bboxes(self, do_use_trOCR:bool, logger:Logger):
        """
        Detect text in the image using the trOCR model with pre-detected Google Vision bounding boxes

        Args:
            do_use_trOCR (bool): Whether to use the trOCR model
            logger (Logger): Logger object

        Returns:
            ocr_parts (str): The text detected by the trOCR model
        """
        CONFIDENCES = 0.80
        MAX_NEW_TOKENS = 50
        
        ocr_parts = ''
        if not do_use_trOCR:
            if 'normal' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_printed'] = self.normal_organized_text
                ocr_parts = self.normal_organized_text
            
            if 'hand' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_handwritten'] = self.hand_organized_text
                ocr_parts = self.hand_organized_text

            return ocr_parts
        else:
            logger.info(f'Supplementing with trOCR')

            self.trOCR_texts = []
            original_image = Image.open(self.path).convert("RGB")

            if 'normal' in self.OCR_option or 'CRAFT' in self.OCR_option:
                available_bounds = [element["vertices"] for element in self.normal_text_to_box_mapping if element["granularity"] == "word"]
            elif 'hand' in self.OCR_option:
                available_bounds = [element["vertices"] for element in self.hand_text_to_box_mapping if element["granularity"] == "word"]
            else:
                # Raise exception that there are no available bounds to process
                raise ValueError("No available bounds for TrOCR to process. Please select a Google OCR or CRAFT method first.")

            # Redo logic to call the Google Vision API to get the bounding boxes if they do not exist when using CRAFT

            text_to_box_mapping = []
            total_b = len(available_bounds)
            i = 0
            for bound in tqdm(available_bounds, desc="Processing words using Google Vision bboxes"):
                i+=1
                if self.json_report:
                    self.json_report.set_text(text_main=f'Working on trOCR :construction: {i}/{total_b}')

                vertices = bound["vertices"]

                left = min([v["x"] for v in vertices])
                top = min([v["y"] for v in vertices])
                right = max([v["x"] for v in vertices])
                bottom = max([v["y"] for v in vertices])

                # Crop image based on Google's bounding box
                cropped_image = original_image.crop((left, top, right, bottom))
                pixel_values = self.trOCR_processor(cropped_image, return_tensors="pt").pixel_values

                # Move pixel values to the appropriate device
                pixel_values = pixel_values.to(self.device)

                generated_ids = self.trOCR_model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
                extracted_text = self.trOCR_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.trOCR_texts.append(extracted_text)

                text_to_box_mapping.append({
                    "vertices": vertices,
                    "text": extracted_text,  # Use the text extracted by trOCR
                    "granularity": "word",
                    "confidence": CONFIDENCES
                    })

            self.trOCR_texts = ' '.join(self.trOCR_texts)
            self.trOCR_text_to_box_mapping = text_to_box_mapping

            if 'normal' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_printed'] = self.normal_organized_text
                self.OCR_JSON_to_file['OCR_trOCR'] = self.trOCR_texts
                ocr_parts = self.trOCR_texts
            if 'hand' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_handwritten'] = self.hand_organized_text
                self.OCR_JSON_to_file['OCR_trOCR'] = self.trOCR_texts
                ocr_parts = self.trOCR_texts
            if 'CRAFT' in self.OCR_option:
                self.OCR_JSON_to_file['OCR_CRAFT_trOCR'] = self.trOCR_texts
                ocr_parts = self.trOCR_texts

            return ocr_parts

    def render_text_on_black_image(self, option):
        '''
        Renders the text extracted by the OCR engine on a black image

        Args:
            option (str): The OCR option to render text for (e.g., 'normal', 'hand', ...)
        '''
        original_image = Image.open(self.path)
        width, height = original_image.size
        black_image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(black_image)

        for element in getattr(self, f'{option}_text_to_box_mapping', []):
            if element["granularity"] != "character":
                continue
            # Determine font size based on the height of the bounding box
            font_size = element["vertices"][2]["y"] - element["vertices"][0]["y"]
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default().font_variant(size=font_size)
            color = OCREngine.confidence_to_color(element["confidence"])
            position = (element["vertices"][0]["x"], element["vertices"][0]["y"] - font_size)
            draw.text(position, element['text'], fill=color, font=font)

        return black_image

    def merge_images(self, image1:Image, image2:Image) -> Image:
        """
        Merge two images side by side

        Args:
            image1 (PIL.Image): First image
            image2 (PIL.Image): Second image

        Returns:
            (PIL.Image): Merged image
        """
        width1, height1 = image1.size
        width2, height2 = image2.size
        merged_image = Image.new("RGB", (width1 + width2, max([height1, height2])))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (width1, 0))
        return merged_image

    def draw_helper_boxes(self, ocr_method, granularity="all"):
        # Raise exception if using granularity control without handwritten or normal selected (Google)
        if ocr_method == "hand" and "hand" not in self.OCR_option or ocr_method == "normal" and "normal" not in self.OCR_option:
            raise ValueError(f"Cannot draw boxes for {ocr_method} OCR as it is not selected. Granularity control is only available for Google OCR, keep default otherwise.")
        # Raise exception if using an invalid granularity
        if granularity not in OCREngine.VALID_GRANULARITIES:
            raise ValueError(f"Invalid granularity '{granularity}'. Valid granularities are: {', '.join(OCREngine.VALID_GRANULARITIES)}")
        
        image_copy = self.image.copy() # Copy the original image to draw on
        draw = ImageDraw.Draw(image_copy)
        width, height = image_copy.size
        text_to_box_mapping = getattr(self, f'{ocr_method}_text_to_box_mapping', [])

        if min([width, height]) > 4000:
            line_width_thick = int((width + height) / 2 * 0.0025)  # Adjust line width for character level
            line_width_thin = 1
        else:
            line_width_thick = int((width + height) / 2 * 0.005)  # Adjust line width for character level
            line_width_thin = 1 #int((width + height) / 2 * 0.001)

        for element in text_to_box_mapping:
            if granularity != "character" and element["granularity"] == "character":
                # Draw lines at the bottom of characters
                bottom_left = (element["vertices"][3]["x"], element["vertices"][3]["y"] + line_width_thick)
                bottom_right = (element["vertices"][2]["x"], element["vertices"][2]["y"] + line_width_thick)
                draw.line([bottom_left, bottom_right], fill=OCREngine.confidence_to_color(element["confidence"]), width=line_width_thick)
            if granularity == "all" or element["granularity"] == granularity:
                # Draw boxes around elements at specified granularity
                vertices = element["vertices"]
                draw.polygon(
                    [
                        vertices[0]["x"], vertices[0]["y"],
                        vertices[1]["x"], vertices[1]["y"],
                        vertices[2]["x"], vertices[2]["y"],
                        vertices[3]["x"], vertices[3]["y"]
                    ],
                    fill=None,
                    outline=OCREngine.confidence_to_color(element["confidence"]),
                    width=line_width_thin
                )
                
        return image_copy
    

    def create_redacted_image(self):
        """
        Draw redaction boxes on the image based on the classified text
        """
        self.logger.info("Drawing redaction boxes")

        image_copy = self.image.copy()
        # self.OCR_option
        if 'normal' in self.OCR_option and 'hand' in self.OCR_option:
            # Merge normal boxes and hand boxes into a single list
            text_to_box_mapping = self.normal_text_to_box_mapping + self.hand_text_to_box_mapping
        elif 'normal' in self.OCR_option:
            text_to_box_mapping = self.normal_text_to_box_mapping
        elif 'hand' in self.OCR_option:
            text_to_box_mapping = self.hand_text_to_box_mapping
        else:
            self.logger.warning("Did not draw redaction boxes as no Google OCR method is selected.")
            return

        draw = ImageDraw.Draw(image_copy)

        for element in text_to_box_mapping:
            if element["locational"] == 1:
                vertices = element["vertices"]
                draw.polygon(
                    [
                        vertices[0]["x"], vertices[0]["y"],
                        vertices[1]["x"], vertices[1]["y"],
                        vertices[2]["x"], vertices[2]["y"],
                        vertices[3]["x"], vertices[3]["y"]
                    ],
                    fill='red',
                    outline=None,
                )

        self.image_redacted = image_copy


    def detect_text(self, handwritten:bool=False):
        """
        Detect handwritten text in the image using Google Cloud Vision API

        Returns:
            (str) self.hand_cleaned_text: Handwritten text detected in the image
        """
        
        with open(self.path, "rb") as image_file:
            content = image_file.read()

        image = vision_beta.Image(content=content)
        image_context = vision_beta.ImageContext(language_hints=["en-t-i0-handwrit"]) if handwritten else None
        response = self.client_beta.document_text_detection(image=image, image_context=image_context)
        
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        """
        text_to_box_mapping element example:
        {
            "vertices": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
            "text": "lorem ipsum",
            "confidence": 0.9,
            "granularity": "line"
            "locational": 0
        }
        """
        text_to_box_mapping = []
        organized_text = ""

        for page in response.full_text_annotation.pages:
            organized_text += self.get_element_text(page) + '\n'
            for block in page.blocks:
                text_to_box_mapping.append({"vertices": self.get_element_vertices(block),
                                            "text": self.get_element_text(block),
                                            "confidence": block.confidence,
                                            "granularity": "block",
                                            "locational": 0})
                for paragraph in block.paragraphs:
                    text_to_box_mapping.append({"vertices": self.get_element_vertices(paragraph),
                                                "text": self.get_element_text(paragraph),
                                                "confidence": paragraph.confidence,
                                                "granularity": "paragraph",
                                                "locational": 0})
                    lines = OCREngine.paragraph_to_lines(paragraph) # Split paragraph to custom 'OCREngine.Lines' granularity
                    for line in lines:
                        text_to_box_mapping.append({"vertices": self.get_element_vertices(line),
                                                    "text": self.get_element_text(line),
                                                    "confidence": line.confidence,
                                                    "granularity": "line",
                                                    "locational": 0})
                        for word in line.words:
                            text_to_box_mapping.append({"vertices": self.get_element_vertices(word),
                                                        "text": self.get_element_text(word),
                                                        "confidence": word.confidence,
                                                        "granularity": "word",
                                                        "locational": 0})


        if handwritten:
            self.hand_cleaned_text = response.text_annotations[0].description if response.text_annotations else '' # Text dump
            self.hand_organized_text = organized_text                                                              # Organized text
            self.hand_text_to_box_mapping = text_to_box_mapping                                                    # Text elements with boxes and granularity
            return self.hand_cleaned_text                                                    
        else:
            self.normal_cleaned_text = response.text_annotations[0].description if response.text_annotations else ''
            self.normal_organized_text = organized_text
            self.normal_text_to_box_mapping = text_to_box_mapping
            return self.normal_cleaned_text

    def classify_text(self, ocr_option) -> None:
        '''
        Access the dictionary of detected OCR text and classify it using the Redactor object depending on the chosen granularity.

        Args:
            ocr_option (str): The OCR option (e.g., 'normal', 'hand', ...)
        '''
        if self.redactor is None:
            self.logger.warning("No redactor provided. Cannot classify text.")
            return # Do not modify mapping if redactor is not provided
        self.logger.info(f"Classifying text for {ocr_option} OCR")
        for element in getattr(self, f"{ocr_option}_text_to_box_mapping", []):
            print("Processing an OCR element...")
            print("element granularity", element["granularity"], type(element["granularity"]))
            print("redaction granularity", self.redaction_granularity, type(self.redaction_granularity))
            if element["granularity"] == self.redaction_granularity or self.redaction_granularity == "all":
                classification, _ = self.redactor.classify(element["text"])
                element["locational"] = classification
                print(element)
            else:
                print("Skipping element as it does not match the redaction granularity")


    def process_image(self, do_create_OCR_helper_image:bool, logger:Logger) -> None:
        '''
        Retrieves text from the image using the selected OCR methods,

        Args:
            do_create_OCR_helper_image (bool): Whether to create a visual representation of OCR results
            logger (Logger): Python Logger object
        '''

        # Helper image can only be created if a Google OCR option is selected
        if 'hand' not in self.OCR_option and 'normal' not in self.OCR_option:
            do_create_OCR_helper_image = False

        self.OCR = 'OCR:\n' # Initialize OCR result string

        if 'CRAFT' in self.OCR_option:
            # CRAFT performs text detection, but not recognition
            self.do_use_trOCR = True
            self.detect_text_craft() # Populates self.normal_text_to_box_mapping with detection regions, but no meaningful text content
            # Use TrOCR to recognize text in the detected regions
            if self.double_OCR:
                part_OCR = "\CRAFT trOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)
                self.OCR = self.OCR + part_OCR + part_OCR
            else:
                self.OCR = self.OCR + "\CRAFT trOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)

        if 'LLaVA' in self.OCR_option: # This option does not produce an OCR helper image
            if self.json_report:
                self.json_report.set_text(text_main=f'Working on LLaVA {self.Llava.model_path} OCR :construction:')

            _, _, _, str_output, usage_report = self.Llava.transcribe_image(self.path, self.multimodal_prompt)
            self.logger.info(f"LLaVA Usage Report for Model {self.Llava.model_path}:\n{usage_report}")

            self.OCR_JSON_to_file['OCR_LLaVA'] = str_output

            if self.double_OCR:
                self.OCR = self.OCR + f"\nLLaVA OCR:\n{str_output}" + f"\nLLaVA OCR:\n{str_output}"
            else:
                self.OCR = self.OCR + f"\nLLaVA OCR:\n{str_output}"

        if 'Florence-2' in self.OCR_option: # This option does not produce an OCR helper image
            if self.json_report:
                self.json_report.set_text(text_main=f'Working on Florence-2 [{self.Florence.model_id}] OCR :construction:')

            self.logger.info(f"Florence-2 Usage Report for Model [{self.Florence.model_id}]")
            results_text, _, _, usage_report = self.Florence.ocr_florence(self.path, task_prompt='<OCR>', text_input=None)

            self.OCR_JSON_to_file['OCR_Florence'] = results_text

            if self.double_OCR:
                self.OCR = self.OCR + f"\nFlorence-2 OCR:\n{results_text}" + f"\nFlorence-2 OCR:\n{results_text}"
            else:
                self.OCR = self.OCR + f"\nFlorence-2 OCR:\n{results_text}"

        if 'GPT-4o-mini' in self.OCR_option: # This option does not produce an OCR helper image
            if self.json_report:
                self.json_report.set_text(text_main=f'Working on GPT-4o-mini OCR :construction:')

            self.logger.info(f"GPT-4o-mini Usage Report")
            results_text, _, _, total_cost, _, _, self.tokens_in, self.tokens_out = self.GPTmini.ocr_gpt4o(self.path, resolution=self.cfg['leafmachine']['project']['OCR_GPT_4o_mini_resolution'], max_tokens=512)
            self.cost += total_cost

            self.OCR_JSON_to_file['OCR_GPT_4o_mini'] = results_text

            if self.double_OCR:
                self.OCR = self.OCR + f"\nGPT-4o-mini OCR:\n{results_text}" + f"\nGPT-4o-mini OCR:\n{results_text}"
            else:
                self.OCR = self.OCR + f"\nGPT-4o-mini OCR:\n{results_text}"

        if 'normal' in self.OCR_option:
            if self.double_OCR:
                part_OCR = self.OCR + "\nGoogle Printed OCR:\n" + self.detect_text()
                self.OCR = self.OCR + part_OCR + part_OCR
            else:
                self.OCR = self.OCR + "\nGoogle Printed OCR:\n" + self.detect_text()

        if 'hand' in self.OCR_option:
            if self.double_OCR:
                part_OCR = self.OCR + "\nGoogle Handwritten OCR:\n" + self.detect_text(handwritten=True)
                self.OCR = self.OCR + part_OCR + part_OCR
            else:
                self.OCR = self.OCR + "\nGoogle Handwritten OCR:\n" + self.detect_text(handwritten=True)

            # Optionally add trOCR to the self.OCR for additional context
            if self.do_use_trOCR:
                if self.double_OCR:
                    part_OCR = "\ntrOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)
                    self.OCR = self.OCR + part_OCR + part_OCR
                else:
                    self.OCR = self.OCR + "\ntrOCR:\n" + self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)
            else:
                _ = self.detect_text_with_trOCR_using_google_bboxes(self.do_use_trOCR, logger)

        if do_create_OCR_helper_image: # Now create helper image if requested
            logger.info(f'Creating OCR helper image')
            self.create_ocr_helper_image()

        if self.redactor is not None:
            if 'normal' in self.OCR_option:
                self.classify_text('normal')
            if 'hand' in self.OCR_option:
                self.classify_text('hand')
            self.create_redacted_image()

    def create_ocr_helper_image(self):
        '''
        Creates a stitched image of various OCR visualizations depending on chosen OCR methods
        '''
        # if do_create_OCR_helper_image and ('LLaVA' not in self.OCR_option):
        self.image = Image.open(self.path)

        if 'normal' in self.OCR_option:
            image_with_boxes_normal = self.draw_helper_boxes('normal')
            text_image_normal = self.render_text_on_black_image('normal')
            self.merged_image_normal = self.merge_images(image_with_boxes_normal, text_image_normal)

        if 'hand' in self.OCR_option:
            image_with_boxes_hand = self.draw_helper_boxes('hand')
            text_image_hand = self.render_text_on_black_image('hand')
            self.merged_image_hand = self.merge_images(image_with_boxes_hand, text_image_hand)

        if self.do_use_trOCR:
            text_image_trOCR = self.render_text_on_black_image('trOCR') 

        if 'CRAFT' in self.OCR_option:
            image_with_boxes_normal = self.draw_helper_boxes('normal')
            self.merged_image_normal = self.merge_images(image_with_boxes_normal, text_image_trOCR)

        # Merge final overlay image [original, normal bboxes, normal text]
        if 'hand' in self.OCR_option or 'normal' in self.OCR_option:
            if 'CRAFT' in self.OCR_option or 'normal' in self.OCR_option:
                self.image_ocr = self.merge_images(Image.open(self.path), self.merged_image_normal)
            elif 'hand' in self.OCR_option: # [original, hand bboxes, hand text]
                self.image_ocr = self.merge_images(Image.open(self.path), self.merged_image_hand)
            else: # [original, normal bboxes, normal text, hand bboxes, hand text]
                self.image_ocr = self.merge_images(Image.open(self.path), self.merge_images(self.merged_image_normal, self.merged_image_hand))
        
        if self.do_use_trOCR:
            if 'CRAFT' in self.OCR_option:
                heat_map_text = Image.fromarray(cv2.cvtColor(self.prediction_result["heatmaps"]["text_score_heatmap"], cv2.COLOR_BGR2RGB))
                heat_map_link = Image.fromarray(cv2.cvtColor(self.prediction_result["heatmaps"]["link_score_heatmap"], cv2.COLOR_BGR2RGB))
                self.image_ocr = self.merge_images(self.image_ocr, heat_map_text)
                self.image_ocr = self.merge_images(self.image_ocr, heat_map_link)
            else:
                self.image_ocr = self.merge_images(self.image_ocr, text_image_trOCR)
    
        try:
            from craft_text_detector import empty_cuda_cache
            empty_cuda_cache()
        except:
            pass


class Redactor():
    """
    Class to redact sensitive information from an image using the RoBERTa model
    """

    def __init__(self, device:str, logger:Logger, banned_words:set|None=None, model_id:str="roberta-base", tokenizer_id:str="roberta-base") -> None:
        """
        Initialize the Redactor object

        Args:
            model (str): Path to the model directory
            tokenizer (str): Path to the tokenizer directory
            device (str): Device to run the model on
            logger (Logger): Logger object
            banned_words (set): Set of banned words
        """

        if model_id in ["roberta-base","roberta-small","roberta-large","roberta-large-mnli"]:
            logger.warning("Using a standard RoBERTa model, accuracy will be poor!\n Please give 'model' argument a fine-tuned model path, not \'roberta-base\'.")
        if banned_words is None:
            logger.warning("No banned words provided. Only classifier model will be used for redaction.")

        self.device = device
        self.logger = logger
        self.banned_words = banned_words
        self.model = RobertaForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_id)

    def classify(self, text:str) -> bool:
        """
        Classify text as sensitive or not

        Args:
            text (str): Text to classify

        Returns:
            (bool): True if text is sensitive, False otherwise
        """
        banned_word_pattern = r'\b(?:' + '|'.join(map(re.escape, self.banned_words)) + r')\b'

        if re.search(banned_word_pattern, text.lower()):
            # If the paragraph contains a banned word, classify as location sensitive
            predicted_class = 1
            confidence = 1.0
        else:
            # Use RoBERTa model to classify text
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1).item()
            print(predicted_class)
            confidence = logits.softmax(dim=-1).max().item()

            self.logger.info(f"Text: {text}")
            self.logger.info(f"Predicted class: {predicted_class}")
            self.logger.info(f"Confidence: {confidence}")
            self.logger.info("**********")
            

        return predicted_class, confidence

    def add_banned_words(self, banned_words:set) -> None:
        """
        Add banned words to the set of banned words

        Args:
            banned_words (set): Set of banned words to add
        """
        self.banned_words.update(banned_words)

    def reset_banned_words(self) -> None:
        """
        Empty the set of banned words
        """
        self.banned_words = set()


class SafetyCheck():
    def __init__(self, is_hf) -> None:
        self.is_hf = is_hf
        self.set_client()

    def set_client(self):
        if self.is_hf:
            self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())
        else:
            self.client = vision.ImageAnnotatorClient(credentials=self.get_google_credentials())


    def get_google_credentials(self):
        creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json_str))
        return credentials
    
    def check_for_inappropriate_content(self, file_stream):
        LEVEL = 2
        content = file_stream.read()
        image = vision.Image(content=content)
        response = self.client.safe_search_detection(image=image)
        safe = response.safe_search_annotation

        likelihood_name = (
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
        )
        print("Safe search:")

        print(f"    adult*: {likelihood_name[safe.adult]}")
        print(f"    medical*: {likelihood_name[safe.medical]}")
        print(f"    spoofed: {likelihood_name[safe.spoof]}")
        print(f"    violence*: {likelihood_name[safe.violence]}")
        print(f"    racy: {likelihood_name[safe.racy]}")

        # Check the levels of adult, violence, racy, etc. content.
        if (safe.adult > LEVEL or
            safe.medical > LEVEL or
            # safe.spoof > LEVEL or
            safe.violence > LEVEL #or
            # safe.racy > LEVEL
            ):
            print("Found violation")
            return True  # The image violates safe search guidelines.

        print("Found NO violation")
        return False  # The image is considered safe.