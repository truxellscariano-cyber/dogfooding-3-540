import os
import pandas as pd
from sentiment_analysis.config import Config
from sentiment_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data() -> pd.DataFrame:
    logger.info("Loading data...")
    if not os.path.exists(Config.DATA_FILE):
        data = {
            'review': [
                'This movie was absolutely fantastic! I loved every minute of it.',
                'Terrible film, waste of time and money.',
                'Great acting and amazing storyline. Highly recommended!',
                'Boring and predictable. Would not watch again.',
                'One of the best movies I have ever seen!',
                'Awful movie with poor acting and bad script.',
                'Excellent cinematography and compelling characters.',
                'Disappointing and overrated. Not worth watching.',
                'Brilliant performance by the lead actor!',
                'Worst movie of the year. Complete disaster.',
            ] * 50,
            'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                         'negative', 'positive', 'negative', 'positive', 'negative'] * 50
        }
        df = pd.DataFrame(data)
        df.to_csv(Config.DATA_FILE, index=False)
        logger.info(f"Created sample dataset: %s", Config.DATA_FILE)
    else:
        df = pd.read_csv(Config.DATA_FILE)
    
    logger.info(f"Loaded %d reviews", len(df))
    return df
