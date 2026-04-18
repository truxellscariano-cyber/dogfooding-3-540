import pandas as pd
import os
from typing import Optional
from ..config import Config
from ..utils.logger import get_logger


class DataLoader:
    def __init__(self, base_dir: str = ''):
        self.base_dir = base_dir
        self.logger = get_logger('sentiment_analysis.data')
        self._data: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        if self._data is not None:
            self.logger.debug("Returning cached data")
            return self._data
        
        data_path = Config.get_data_path(self.base_dir)
        self.logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            self.logger.warning(f"Data file not found at {data_path}, creating sample data")
            df = self._create_sample_data(data_path)
        else:
            df = pd.read_csv(data_path)
        
        self.logger.info(f"Loaded {len(df)} reviews")
        self._data = df
        return df
    
    def _create_sample_data(self, save_path: str) -> pd.DataFrame:
        reviews = [
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
        ]
        
        sentiments = ['positive', 'negative', 'positive', 'negative', 'positive',
                      'negative', 'positive', 'negative', 'positive', 'negative']
        
        data = {
            'review': reviews * Config.SAMPLE_DATA_SIZE,
            'sentiment': sentiments * Config.SAMPLE_DATA_SIZE
        }
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Created sample dataset: {save_path}")
        
        return df
    
    def clear_cache(self) -> None:
        self._data = None
        self.logger.debug("Data cache cleared")
