import logging

def setup_logger():
    logger = logging.getLogger("RecommendationSystem")
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler("recommendation_system.log", mode='w') 
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger