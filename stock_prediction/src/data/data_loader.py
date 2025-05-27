from tiingo import TiingoClient
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.client = TiingoClient({'api_key': config['data']['api_key']})
    
    def load_data(self):
        """
        Charge les données depuis Tiingo Pro ou un fichier local
        
        Args:
            custom_data_path (str, optional): Chemin vers des données personnalisées
                
        Returns:
            bool: True si les données ont été chargées avec succès
        """
        try:
            import pandas as pd
            import time
            
            # Si un chemin de données personnalisées est fourni, essayer de charger depuis ce chemin
            if custom_data_path:
                try:
                    self.logger.info(f"Tentative de chargement depuis {custom_data_path}")
                    
                    if custom_data_path.endswith('.csv'):
                        df = pd.read_csv(custom_data_path, index_col=0, parse_dates=True)
                        
                        # Vérifier si l'index est correctement formaté comme datetime
                        if not isinstance(df.index, pd.DatetimeIndex):
                            self.logger.warning("L'index n'est pas un DatetimeIndex, conversion...")
                            df.index = pd.to_datetime(df.index)
                            
                        # Trier par ordre chronologique croissant
                        df = df.sort_index()
                        
                        self.data = df
                        self.logger.info(f"Données chargées avec succès depuis {custom_data_path}: {self.data.shape}")
                        return True
                    else:
                        self.logger.error(f"Format de fichier non pris en charge: {custom_data_path}")
                
                except Exception as load_error:
                    self.logger.error(f"Erreur lors du chargement des données depuis {custom_data_path}: {str(load_error)}")
                    self.logger.warning("Tentative de chargement depuis Tiingo...")
            
            # Vérifier si la configuration Tiingo existe
            if 'tiingo' not in self.config or 'api_key' not in self.config['tiingo'] or not self.config['tiingo']['api_key']:
                self.logger.error("Clé API Tiingo manquante. Veuillez configurer une clé API valide.")
                return False
            
            api_key = self.config['tiingo']['api_key']
            if api_key == "VOTRE_CLÉ_API_TIINGO_ICI":
                self.logger.error("Clé API Tiingo non configurée. Veuillez remplacer la valeur par défaut par une clé API réelle.")
                return False
            
            self.logger.info("Chargement des données depuis Tiingo Pro")
            
            # Configuration de Tiingo
            from tiingo import TiingoClient
            
            tiingo_config = {
                'api_key': api_key,
                'session': True
            }
            
            # Initialiser le client Tiingo
            try:
                client = TiingoClient(tiingo_config)
            except Exception as tiingo_error:
                self.logger.error(f"Erreur lors de l'initialisation du client Tiingo: {str(tiingo_error)}")
                return False
            
            # Symboles à récupérer
            symbols = self.config['data']['symbols']
            start_date = pd.to_datetime(self.config['data']['start_date'])
            end_date = pd.to_datetime(self.config['data']['end_date'])
            
            # Format de la date pour Tiingo (YYYY-MM-DD)
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Création d'un dictionnaire pour stocker les DataFrames
            all_data = {}
            
            for symbol in symbols:
                self.logger.info(f"Récupération des données pour {symbol}")
                
                try:
                    # Récupérer les données pour un seul symbole
                    historical_data = client.get_dataframe(
                        symbol,  # IMPORTANT: utiliser un seul symbole à la fois, pas une liste
                        frequency='daily',
                        startDate=start_date_str,
                        endDate=end_date_str
                    )
                    
                    # Vérifier si des données ont été récupérées
                    if historical_data.empty:
                        self.logger.warning(f"Aucune donnée récupérée pour {symbol}")
                        continue
                    
                    # Traiter les données
                    df = historical_data.copy()
                    
                    # Renommer les colonnes pour correspondre au format attendu
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'adjClose': 'Adj Close',
                        'adjVolume': 'Adj Volume',
                        'adjHigh': 'Adj High',
                        'adjLow': 'Adj Low',
                        'adjOpen': 'Adj Open',
                        'divCash': 'Dividends',
                        'splitFactor': 'Split'
                    }
                    
                    # Renommer uniquement les colonnes qui existent
                    for old_name, new_name in column_mapping.items():
                        if old_name in df.columns:
                            df.rename(columns={old_name: new_name}, inplace=True)
                    
                    # Si Adj Close n'existe pas, utilisez Close
                    if 'Adj Close' not in df.columns and 'Close' in df.columns:
                        df['Adj Close'] = df['Close']
                    
                    # Gérer les problèmes de fuseau horaire dans l'index
                    if hasattr(df.index, 'tz') and df.index.tz is not None:
                        # Supprimer l'information de fuseau horaire
                        df.index = df.index.tz_localize(None)
                    
                    # S'assurer que les dates de début et de fin sont également sans fuseau horaire
                    start_date_naive = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None else start_date
                    end_date_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None else end_date
                    
                    # Filtrer selon les dates
                    try:
                        df = df.loc[(df.index >= start_date_naive) & (df.index <= end_date_naive)]
                    except TypeError as type_error:
                        # Si TypeError persiste, essayons une autre approche
                        self.logger.warning(f"Erreur de type lors du filtrage des dates pour {symbol}: {str(type_error)}")
                        # Convertir l'index en chaîne et comparer les dates sous forme de chaîne
                        date_mask = (df.index.strftime('%Y-%m-%d') >= start_date_naive.strftime('%Y-%m-%d')) & \
                                    (df.index.strftime('%Y-%m-%d') <= end_date_naive.strftime('%Y-%m-%d'))
                        df = df.loc[date_mask]
                    
                    # S'assurer que l'index a une fréquence
                    try:
                        # Pour les données boursières, généralement 'B' (jours ouvrables)
                        df = df.asfreq('B')
                    except ValueError:
                        # Si problème avec asfreq, continuer sans définir la fréquence
                        pass
                    
                    if not df.empty:
                        all_data[symbol] = df
                        self.logger.info(f"Données récupérées pour {symbol}: {df.shape}")
                    else:
                        self.logger.warning(f"Aucune donnée dans la plage de dates pour {symbol}")
                    
                    # Pause pour ne pas surcharger l'API
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
                    continue
            
            # Si aucune donnée n'a été récupérée
            if not all_data:
                self.logger.error("Aucune donnée n'a pu être récupérée depuis Tiingo")
                return False
            
            # Créer un DataFrame multi-indexé à partir des données récupérées
            dfs = []
            for symbol, df in all_data.items():
                # Ajouter le symbole comme niveau dans les colonnes
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                dfs.append(df)
            
            # Concaténer tous les DataFrames
            try:
                self.data = pd.concat(dfs, axis=1)
                
                # Afficher des informations sur les données chargées
                self.logger.info(f"Données chargées avec succès: {self.data.shape}")
                self.logger.info(f"Période des données: {self.data.index[0]} à {self.data.index[-1]}")
                self.logger.info(f"Nombre de lignes: {len(self.data)}")
                
                return True
            except Exception as concat_error:
                self.logger.error(f"Erreur lors de la concaténation des données: {str(concat_error)}")
                return False
            
        except Exception as e:
            self.logger.error(f"Erreur générale lors du chargement des données: {str(e)}")
            traceback.print_exc()
            return False
        
    def get_price_column(self, df, symbol, column_type='Close'):
        """
        Récupère une colonne de prix en gérant les différents formats d'index
        
        Args:
            symbol (str): Symbole boursier
            column_type (str): Type de colonne ('Close', 'Open', etc.)
        
        Returns:
            pd.Series: La colonne de prix demandée ou None si non trouvée
        """
        try:
            # Essayer tous les formats possibles de colonnes
            column_formats = [
                f"{symbol}.{column_type}",           # Format "AAPL.Close"
                (symbol, column_type),               # Format MultiIndex ('AAPL', 'Close')
                f"{symbol}_{column_type}",           # Format "AAPL_Close"
                f"{column_type}_{symbol}",           # Format "Close_AAPL"
                column_type                          # Si une seule série avec juste le type de colonne
            ]
            
            # Vérifier chaque format
            for col_format in column_formats:
                if isinstance(col_format, tuple) and isinstance(self.data.columns, pd.MultiIndex):
                    # Format MultiIndex
                    if col_format in self.data.columns:
                        return self.data[col_format]
                elif col_format in self.data.columns:
                    # Format string
                    return self.data[col_format]
            
            # Si aucun format standard n'a fonctionné, rechercher par correspondance partielle
            for col in self.data.columns:
                col_str = str(col)
                if symbol in col_str and column_type in col_str:
                    self.logger.info(f"Colonne trouvée par correspondance partielle: {col}")
                    return self.data[col]
            
            # Si toujours pas trouvé, lever une exception personnalisée
            raise ValueError(f"Impossible de trouver une colonne de {column_type} pour {symbol}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la colonne {column_type} pour {symbol}: {str(e)}")
            return None