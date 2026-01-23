import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json

class SmartImageCaptioning:
    """Smart image captioning using ImageNet predictions"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V1').to(device)
        self.model.eval()
        
        # ImageNet class names (simplified version)
        self.class_names = self._get_imagenet_classes()
        
        # Category mappings
        self.category_map = {
            'animals': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe'],
            'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane'],
            'people': ['person', 'man', 'woman', 'child', 'baby'],
            'nature': ['tree', 'flower', 'grass', 'mountain', 'beach', 'ocean', 'sky', 'cloud'],
            'objects': ['chair', 'table', 'book', 'phone', 'computer', 'bottle', 'cup', 'plate'],
            'food': ['apple', 'banana', 'pizza', 'cake', 'bread', 'sandwich', 'fruit', 'vegetable']
        }
        
    def _get_imagenet_classes(self):
        """Get simplified ImageNet class names"""
        # Simplified version of common ImageNet classes
        classes = [
            'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray',
            'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco',
            'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel',
            'kite', 'bald_eagle', 'vulture', 'great_grey_owl', 'European_fire_salamander',
            'common_newt', 'eft', 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog',
            'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin',
            'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail',
            'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 'green_lizard',
            'African_chameleon', 'Komodo_dragon', 'African_crocodile', 'American_alligator',
            'triceratops', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake',
            'king_snake', 'garter_snake', 'water_snake', 'vine_snake', 'night_snake',
            'boa_constrictor', 'rock_python', 'Indian_cobra', 'green_mamba', 'sea_snake',
            'horned_viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion',
            'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 'black_widow',
            'tarantula', 'wolf_spider', 'tick', 'centipede', 'black_grouse', 'ptarmigan',
            'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'African_grey',
            'macaw', 'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 'hornbill',
            'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 'goose',
            'black_swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat',
            'jellyfish', 'sea_anemone', 'brain_coral', 'flatworm', 'nematode', 'conch',
            'snail', 'slug', 'sea_slug', 'chiton', 'chambered_nautilus', 'Dungeness_crab',
            'rock_crab', 'fiddler_crab', 'king_crab', 'American_lobster', 'spiny_lobster',
            'crayfish', 'hermit_crab', 'isopod', 'white_stork', 'black_stork', 'spoonbill',
            'flamingo', 'little_blue_heron', 'American_egret', 'bittern', 'crane', 'limpkin',
            'European_gallinule', 'American_coot', 'bustard', 'ruddy_turnstone', 'red-backed_sandpiper',
            'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king_penguin', 'albatross',
            'grey_whale', 'killer_whale', 'dugong', 'sea_lion', 'Chihuahua', 'Japanese_spaniel',
            'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier',
            'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
            'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi',
            'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound',
            'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier',
            'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier',
            'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier',
            'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier',
            'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer',
            'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
            'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
            'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
            'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel',
            'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel',
            'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard',
            'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie',
            'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman',
            'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller',
            'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane',
            'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'dalmatian', 'affenpinscher',
            'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian',
            'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle',
            'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'timber_wolf', 'white_wolf',
            'red_wolf', 'coyote', 'dingo', 'dhole', 'African_hunting_dog', 'hyena', 'red_fox',
            'kit_fox', 'Arctic_fox', 'grey_fox', 'tabby', 'tiger_cat', 'Persian_cat',
            'Siamese_cat', 'Egyptian_cat', 'cougar', 'lynx', 'leopard', 'snow_leopard',
            'jaguar', 'lion', 'tiger', 'cheetah', 'brown_bear', 'American_black_bear',
            'ice_bear', 'sloth_bear', 'mongoose', 'meerkat', 'tiger_beetle', 'ladybug',
            'ground_beetle', 'long-horned_beetle', 'leaf_beetle', 'dung_beetle', 'rhinoceros_beetle',
            'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking_stick',
            'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly',
            'admiral', 'ringlet', 'monarch', 'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid',
            'starfish', 'sea_urchin', 'sea_cucumber', 'wood_rabbit', 'hare', 'Angora',
            'hamster', 'porcupine', 'fox_squirrel', 'marmot', 'beaver', 'guinea_pig',
            'sorrel', 'zebra', 'hog', 'wild_boar', 'warthog', 'hippopotamus', 'ox',
            'water_buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala',
            'gazelle', 'Arabian_camel', 'llama', 'weasel', 'mink', 'polecat', 'black-footed_ferret',
            'otter', 'skunk', 'badger', 'armadillo', 'three-toed_sloth', 'orangutan', 'gorilla',
            'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas', 'baboon', 'macaque',
            'langur', 'colobus', 'proboscis_monkey', 'marmoset', 'capuchin', 'howler_monkey',
            'titi', 'spider_monkey', 'squirrel_monkey', 'Madagascar_cat', 'indri', 'Indian_elephant',
            'African_elephant', 'lesser_panda', 'giant_panda', 'barracouta', 'eel', 'coho',
            'rock_beauty', 'anemone_fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus',
            'abaya', 'academic_gown', 'accordion', 'acoustic_guitar', 'aircraft_carrier', 'airliner',
            'airship', 'altar', 'ambulance', 'amphibian', 'analog_clock', 'apiary', 'apron',
            'ashcan', 'assault_rifle', 'backpack', 'bakery', 'balance_beam', 'balloon', 'ballpoint',
            'Band_Aid', 'banjo', 'bannister', 'barbell', 'barber_chair', 'barbershop', 'barn',
            'barometer', 'barrel', 'barrow', 'baseball', 'basketball', 'bassinet', 'bassoon',
            'bathing_cap', 'bath_towel', 'bathtub', 'beach_wagon', 'beacon', 'beaker', 'bearskin',
            'beer_bottle', 'beer_glass', 'bell_cote', 'bib', 'bicycle-built-for-two', 'bikini',
            'binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo_tie', 'bonnet',
            'bookcase', 'bookshop', 'bottlecap', 'bow', 'bow_tie', 'brass', 'brassiere',
            'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof_vest',
            'bullet_train', 'butcher_shop', 'cab', 'caldron', 'candle', 'cannon', 'canoe',
            'can_opener', 'cardigan', 'car_mirror', 'carousel', 'carpenter_kit', 'carton',
            'car_wheel', 'cash_machine', 'cassette', 'cassette_player', 'castle', 'catamaran',
            'CD_player', 'cello', 'cellular_telephone', 'chain', 'chainlink_fence', 'chain_mail',
            'chain_saw', 'chest', 'chiffonier', 'chime', 'china_cabinet', 'Christmas_stocking',
            'church', 'cinema', 'cleaver', 'cliff_dwelling', 'cloak', 'clog', 'cocktail_shaker',
            'coffee_mug', 'coffeepot', 'coil', 'combination_lock', 'computer_keyboard', 'confectionery',
            'container_ship', 'convertible', 'corkscrew', 'cornet', 'cowboy_boot', 'cowboy_hat',
            'cradle', 'crane', 'crash_helmet', 'crate', 'crib', 'Crock_Pot', 'croquet_ball',
            'crutch', 'cuirass', 'dam', 'desk', 'desktop_computer', 'dial_telephone', 'diaper',
            'digital_clock', 'digital_watch', 'dining_table', 'dishrag', 'dishwasher', 'disk_brake',
            'dock', 'dogsled', 'dome', 'doormat', 'drilling_platform', 'drum', 'drumstick',
            'dumbbell', 'Dutch_oven', 'electric_fan', 'electric_guitar', 'electric_locomotive',
            'entertainment_center', 'envelope', 'espresso_maker', 'face_powder', 'feather_boa',
            'file', 'fireboat', 'fire_engine', 'fire_screen', 'flagpole', 'flute', 'folding_chair',
            'football_helmet', 'forklift', 'fountain', 'fountain_pen', 'four-poster', 'freight_car',
            'French_horn', 'frying_pan', 'fur_coat', 'garbage_truck', 'gasmask', 'gas_pump',
            'goblet', 'go-kart', 'golf_ball', 'golfcart', 'gondola', 'gong', 'gown', 'grand_piano',
            'greenhouse', 'grille', 'grocery_store', 'guillotine', 'hair_slide', 'hair_spray',
            'half_track', 'hammer', 'hamper', 'hand_blower', 'hand-held_computer', 'handkerchief',
            'hard_disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home_theater',
            'honeycomb', 'hook', 'hoopskirt', 'horizontal_bar', 'horse_cart', 'hourglass',
            'iPod', 'iron', 'jack-o-lantern', 'jean', 'jeep', 'jersey', 'jigsaw_puzzle',
            'jinrikisha', 'joystick', 'kimono', 'knee_pad', 'knot', 'lab_coat', 'ladle',
            'lampshade', 'laptop', 'lawn_mower', 'lens_cap', 'letter_opener', 'library',
            'lifeboat', 'lighter', 'limousine', 'liner', 'lipstick', 'Loafer', 'lotion',
            'loudspeaker', 'loupe', 'lumbermill', 'magnetic_compass', 'mailbag', 'mailbox',
            'maillot', 'maillot', 'manhole_cover', 'maraca', 'marimba', 'mask', 'matchstick',
            'maypole', 'maze', 'measuring_cup', 'medicine_chest', 'megalith', 'microphone',
            'microwave', 'military_uniform', 'milk_can', 'minibus', 'miniskirt', 'minivan',
            'missile', 'mitten', 'mixing_bowl', 'mobile_home', 'Model_T', 'modem', 'monastery',
            'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito_net', 'motor_scooter',
            'mountain_bike', 'mountain_tent', 'mouse', 'mousetrap', 'moving_van', 'muzzle',
            'nail', 'neck_brace', 'necklace', 'nipple', 'notebook', 'obelisk', 'oboe',
            'ocarina', 'odometer', 'oil_filter', 'organ', 'oscilloscope', 'overskirt', 'oxcart',
            'oxygen_mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama',
            'palace', 'panpipe', 'paper_towel', 'parachute', 'parallel_bars', 'park_bench',
            'parking_meter', 'passenger_car', 'patio', 'pay-phone', 'pedestal', 'pencil_box',
            'pencil_sharpener', 'perfume', 'Petri_dish', 'photocopier', 'pick', 'pickelhaube',
            'picket_fence', 'pickup', 'pier', 'piggy_bank', 'pill_bottle', 'pillow', 'ping-pong_ball',
            'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic_bag', 'plate_rack',
            'plow', 'plunger', 'Polaroid_camera', 'pole', 'police_van', 'poncho', 'pool_table',
            'pop_bottle', 'pot', 'potter_wheel', 'power_drill', 'prayer_rug', 'printer',
            'prison', 'projectile', 'projector', 'puck', 'punching_bag', 'purse', 'quill',
            'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio_telescope', 'rain_barrel',
            'recreational_vehicle', 'reel', 'reflex_camera', 'refrigerator', 'remote_control',
            'restaurant', 'revolver', 'rifle', 'rocking_chair', 'rotisserie', 'rubber_eraser',
            'rugby_ball', 'rule', 'running_shoe', 'safe', 'safety_pin', 'saltshaker', 'sandal',
            'sarong', 'sax', 'scabbard', 'scale', 'school_bus', 'schooner', 'scoreboard',
            'screen', 'screw', 'screwdriver', 'seat_belt', 'sewing_machine', 'shield', 'shoe_shop',
            'shoji', 'shopping_basket', 'shopping_cart', 'shovel', 'shower_cap', 'shower_curtain',
            'ski', 'ski_mask', 'sleeping_bag', 'slide_rule', 'sliding_door', 'slot', 'snorkel',
            'snowmobile', 'snowplow', 'soap_dispenser', 'soccer_ball', 'sock', 'solar_dish',
            'sombrero', 'soup_bowl', 'space_bar', 'space_heater', 'space_shuttle', 'spatula',
            'speedboat', 'spider_web', 'spindle', 'sports_car', 'spotlight', 'stage', 'steam_locomotive',
            'steel_arch_bridge', 'steel_drum', 'stethoscope', 'stole', 'stone_wall', 'stopwatch',
            'stove', 'strainer', 'streetcar', 'stretcher', 'studio_couch', 'stupa', 'submarine',
            'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension_bridge',
            'swab', 'sweatshirt', 'swimming_trunks', 'swing', 'switch', 'syringe', 'table_lamp',
            'tank', 'tape_player', 'teapot', 'teddy', 'television', 'tennis_ball', 'thatch',
            'theater_curtain', 'thimble', 'thresher', 'throne', 'thumb_tack', 'tiara', 'tiger_beetle',
            'tights', 'till', 'toaster', 'tobacco_shop', 'toilet_seat', 'torch', 'totem_pole',
            'tow_truck', 'toyshop', 'tractor', 'trailer_truck', 'tray', 'trench_coat', 'tricycle',
            'trimaran', 'tripod', 'triumphal_arch', 'trolleybus', 'trombone', 'tub', 'turnstile',
            'typewriter_keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'vase', 'vault',
            'velvet', 'vending_machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle_iron',
            'wall_clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water_bottle',
            'water_jug', 'water_tower', 'whiskey_jug', 'whistle', 'wig', 'window_screen',
            'window_shade', 'Windsor_tie', 'wine_bottle', 'wing', 'wok', 'wooden_spoon',
            'wool', 'worm_fence', 'wreck', 'yawl', 'yurt', 'web_site', 'comic_book',
            'crossword_puzzle', 'street_sign', 'traffic_light', 'book_jacket', 'menu', 'plate',
            'guacamole', 'consomme', 'hot_pot', 'trifle', 'ice_cream', 'ice_lolly', 'French_loaf',
            'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed_potato', 'head_cabbage',
            'broccoli', 'cauliflower', 'zucchini', 'spaghetti_squash', 'acorn_squash', 'butternut_squash',
            'cucumber', 'artichoke', 'bell_pepper', 'cardoon', 'mushroom', 'Granny_Smith',
            'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
            'custard_apple', 'pomegranate', 'hay', 'carbonara', 'chocolate_sauce', 'dough',
            'meat_loaf', 'pizza', 'potpie', 'burrito', 'red_wine', 'espresso', 'cup',
            'eggnog', 'alp', 'bubble', 'cliff', 'coral_reef', 'geyser', 'lakeside',
            'promontory', 'sandbar', 'seashore', 'valley', 'volcano', 'ballplayer', 'groom',
            'scuba_diver', 'rapeseed', 'daisy', 'yellow_lady_slipper', 'corn', 'acorn',
            'hip', 'buckeye', 'coral_fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar',
            'hen-of-the-woods', 'bolete', 'ear', 'toilet_tissue'
        ]
        return classes
    
    def load_image(self, image_path_or_url):
        """Load image from file path or URL"""
        try:
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def predict_classes(self, image):
        """Predict top classes for the image"""
        if image is None:
            return []
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(top5_prob.size(0)):
            class_idx = top5_catid[i].item()
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
                confidence = top5_prob[i].item()
                predictions.append((class_name, confidence))
        
        return predictions
    
    def categorize_predictions(self, predictions):
        """Categorize predictions into semantic groups"""
        categories = {cat: [] for cat in self.category_map.keys()}
        
        for class_name, confidence in predictions:
            # Clean class name
            clean_name = class_name.replace('_', ' ').lower()
            
            # Check each category
            for category, keywords in self.category_map.items():
                for keyword in keywords:
                    if keyword in clean_name or clean_name in keyword:
                        categories[category].append((clean_name, confidence))
                        break
        
        return categories
    
    def generate_caption(self, image_path_or_url):
        """Generate intelligent caption based on ImageNet predictions"""
        image = self.load_image(image_path_or_url)
        if image is None:
            return "unable to process image"
        
        # Get predictions
        predictions = self.predict_classes(image)
        if not predictions:
            return "unable to classify image content"
        
        # Categorize predictions
        categories = self.categorize_predictions(predictions)
        
        # Generate caption based on detected categories
        caption_parts = []
        
        # Check for people
        if categories['people']:
            person_class, confidence = categories['people'][0]
            if confidence > 0.1:
                caption_parts.append(f"a {person_class}")
        
        # Check for animals
        if categories['animals']:
            animal_class, confidence = categories['animals'][0]
            if confidence > 0.1:
                if caption_parts:
                    caption_parts.append(f"with a {animal_class}")
                else:
                    caption_parts.append(f"a {animal_class}")
        
        # Check for vehicles
        if categories['vehicles']:
            vehicle_class, confidence = categories['vehicles'][0]
            if confidence > 0.1:
                if caption_parts:
                    caption_parts.append(f"near a {vehicle_class}")
                else:
                    caption_parts.append(f"a {vehicle_class}")
        
        # Check for nature
        if categories['nature']:
            nature_class, confidence = categories['nature'][0]
            if confidence > 0.1:
                if caption_parts:
                    caption_parts.append(f"in a {nature_class} setting")
                else:
                    caption_parts.append(f"a {nature_class} scene")
        
        # Check for objects
        if categories['objects']:
            object_class, confidence = categories['objects'][0]
            if confidence > 0.1:
                if caption_parts:
                    caption_parts.append(f"with a {object_class}")
                else:
                    caption_parts.append(f"a {object_class}")
        
        # Check for food
        if categories['food']:
            food_class, confidence = categories['food'][0]
            if confidence > 0.1:
                if caption_parts:
                    caption_parts.append(f"featuring {food_class}")
                else:
                    caption_parts.append(f"{food_class}")
        
        # Fallback to top prediction
        if not caption_parts:
            top_class, confidence = predictions[0]
            clean_name = top_class.replace('_', ' ').lower()
            caption_parts.append(f"an image showing {clean_name}")
        
        # Combine parts into a natural sentence
        caption = ' '.join(caption_parts)
        
        # Add confidence information for top prediction
        top_class, top_confidence = predictions[0]
        if top_confidence > 0.5:
            caption += f" (high confidence: {top_confidence:.2f})"
        elif top_confidence > 0.2:
            caption += f" (moderate confidence: {top_confidence:.2f})"
        else:
            caption += f" (low confidence: {top_confidence:.2f})"
        
        return caption
    
    def visualize_result(self, image_path_or_url, caption, predictions=None):
        """Display image with generated caption and predictions"""
        image = self.load_image(image_path_or_url)
        if image is None:
            print("Could not load image for visualization")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Main image
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated Caption: {caption}", fontsize=12, pad=20)
        
        # Predictions bar chart
        if predictions:
            plt.subplot(2, 1, 2)
            classes = [pred[0].replace('_', ' ') for pred in predictions[:5]]
            confidences = [pred[1] for pred in predictions[:5]]
            
            plt.barh(classes, confidences)
            plt.xlabel('Confidence')
            plt.title('Top 5 Predictions')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()

def demo_smart_captioning():
    """Run smart captioning demo"""
    print("Smart Image Captioning Demo")
    print("=" * 40)
    
    # Initialize system
    print("Initializing smart captioning system...")
    captioner = SmartImageCaptioning()
    
    # Test images
    test_images = [
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500",
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500"
    ]
    
    for i, img_url in enumerate(test_images):
        print(f"\nProcessing Image {i+1}...")
        print("-" * 30)
        
        try:
            # Get predictions first
            image = captioner.load_image(img_url)
            if image:
                predictions = captioner.predict_classes(image)
                print("Top predictions:")
                for j, (class_name, confidence) in enumerate(predictions[:3]):
                    print(f"  {j+1}. {class_name.replace('_', ' ')}: {confidence:.3f}")
                
                # Generate caption
                caption = captioner.generate_caption(img_url)
                print(f"\nGenerated Caption: {caption}")
                
                # Visualize (optional)
                show_image = input("Show detailed visualization? (y/n): ").strip().lower()
                if show_image == 'y':
                    captioner.visualize_result(img_url, caption, predictions)
            else:
                print("Could not load image")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nSmart demo completed!")

def interactive_smart_demo():
    """Interactive smart demo"""
    print("Interactive Smart Image Captioning")
    print("=" * 40)
    
    captioner = SmartImageCaptioning()
    
    while True:
        print("\nOptions:")
        print("1. Enter image URL")
        print("2. Exit")
        
        choice = input("Choose option (1-2): ").strip()
        
        if choice == '1':
            url = input("Enter image URL: ").strip()
            if url:
                try:
                    image = captioner.load_image(url)
                    if image:
                        predictions = captioner.predict_classes(image)
                        caption = captioner.generate_caption(url)
                        
                        print(f"\nGenerated Caption: {caption}")
                        print("\nTop predictions:")
                        for j, (class_name, confidence) in enumerate(predictions[:5]):
                            print(f"  {j+1}. {class_name.replace('_', ' ')}: {confidence:.3f}")
                        
                        show = input("Show visualization? (y/n): ").strip().lower()
                        if show == 'y':
                            captioner.visualize_result(url, caption, predictions)
                    else:
                        print("Could not load image")
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == '2':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Smart Image Captioning Demo Options:")
    print("1. Basic smart demo")
    print("2. Interactive demo")
    
    choice = input("\nSelect demo (1-2): ").strip()
    
    if choice == '1':
        demo_smart_captioning()
    elif choice == '2':
        interactive_smart_demo()
    else:
        print("Invalid choice. Running basic demo...")
        demo_smart_captioning()