BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7
MIN_PRICE = 0.5
MAX_PRICE = 999.49
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CATEGORY = [
    "Automotive",
    # "Electronics",
    # "Office_Products",
    # "Tools_and_Home_Improvement",
    # "Cell_Phones_and_Accessories",
    # "Toys_and_Games",
    # "Appliances",
    # "Musical_Instruments",
]
CURATED_DATASET_NAME = "alistermarc/pricing-fine-tuning-curated"
HF_USER = "alistermarc" 
PROJECT_NAME = "alistermarc"
