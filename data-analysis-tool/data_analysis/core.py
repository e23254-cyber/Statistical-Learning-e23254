import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from google.colab import files
from IPython.display import display, HTML

class PlottingMethods:
    """Handles granular chart generation returning HTML dictionaries for Colab display."""
    
    def get_methods_info(self):
        return {"response": ["plot_bar_chart", "plot_pie_chart", "plot_histogram", "display_image"]}

    def plot_bar_chart(self, x, y, data, color=None, barmode='group'):
        fig = px.bar(data, x=x, y=y, color=color, barmode=barmode)
        return {"html": fig.to_html(full_html=False), "status": "success"}

    def plot_pie_chart(self, names, values, data, hole=0.0, title=None):
        fig = px.pie(data, names=names, values=values, hole=hole, title=title)
        return {"html": fig.to_html(full_html=False), "status": "success"}

    def plot_histogram(self, x, data, bins=None, title=None):
        fig = px.histogram(data, x=x, nbins=len(bins)-1 if bins else None, title=title)
        return {"html": fig.to_html(full_html=False), "status": "success"}

    def display_image(self, result):
        if result and result.get('status') == 'success':
            display(HTML(result['html']))


class DataInspector:
    """End-to-end toolkit for CSV ingestion, cleaning, and EDA."""
    
    def __init__(self):
        self.df = None
        self.plotter = PlottingMethods()
        self.encoded_categorical = None
        self.scaled_numeric = None

    # --- 1. Data Ingestion & Sanitization ---
    def upload_data(self):
        print("Please upload your CSV file...")
        uploaded = files.upload()
        for filename in uploaded.keys():
            # Handle garbage strings automatically
            na_values = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
            self.df = pd.read_csv(filename, na_values=na_values)
            self._auto_convert_types()
            print(f"Successfully loaded: {filename}")
            break # Load first file only
            
    def _auto_convert_types(self):
        """Forces conversion to numeric if it doesn't result in an entirely null column."""
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if not converted.isna().all():
                self.df[col] = converted

    # --- 2. Structural Analysis & Cleaning ---
    def get_summary(self):
        print(f"Dimensions: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        display(self.df.head(20))

    def handle_missing_values(self, strategy='median', constant_val=None):
        num_cols = self.df.select_dtypes(include=np.number).columns
        cat_cols = self.df.select_dtypes(exclude=np.number).columns
        
        for col in self.df.columns:
            if strategy == 'mean' and col in num_cols:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median' and col in num_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == 'constant' and constant_val is not None:
                self.df[col].fillna(constant_val, inplace=True)
        print(f"Missing values imputed using '{strategy}' strategy.")

    def remove_duplicates(self):
        initial = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns, find_and_delete=True):
        """IQR-based outlier management."""
        for col in columns:
            if col in self.df.select_dtypes(include=np.number).columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"Found {len(outliers)} outliers in {col}.")
                
                if find_and_delete:
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        if find_and_delete:
            print("Outliers removed.")

    # --- 3. Feature Engineering Preparation ---
    def extract_normalized_numeric_data(self, method='standard'):
        num_df = self.df.select_dtypes(include=np.number)
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
            
        scaled_data = scaler.fit_transform(num_df)
        self.scaled_numeric = pd.DataFrame(scaled_data, columns=num_df.columns)
        return self.scaled_numeric

    def extract_normalized_categorical_data(self, method='onehot'):
        cat_df = self.df.select_dtypes(exclude=np.number)
        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = encoder.fit_transform(cat_df)
            self.encoded_categorical = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        elif method == 'ordinal':
            encoder = OrdinalEncoder()
            encoded_data = encoder.fit_transform(cat_df)
            self.encoded_categorical = pd.DataFrame(encoded_data, columns=cat_df.columns)
        return self.encoded_categorical

    def create_normalized_data_df(self):
        if self.scaled_numeric is not None and self.encoded_categorical is not None:
            return pd.concat([self.scaled_numeric, self.encoded_categorical], axis=1)
        return self.df

    # --- 4. Interactive Visualizations ---
    def plot_numerical(self, column_names):
        """Generates 3-panel subplots (Violin, Scatter, Histogram) for numerics."""
        for col in column_names:
            if col in self.df.columns:
                fig = go.Figure()
                fig.add_trace(go.Violin(x=self.df[col], name='Violin', side='positive'))
                fig.add_trace(go.Scatter(y=self.df[col], mode='markers', name='Scatter'))
                fig.add_trace(go.Histogram(x=self.df[col], name='Histogram'))
                fig.update_layout(title=f'Distribution Analysis: {col}')
                fig.show()

    def plot_relationship(self, col1, col2):
        """Smart relationship chart selection."""
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f'{col1} vs {col2}')
        elif (is_num1 and not is_num2) or (not is_num1 and is_num2):
            x, y = (col2, col1) if is_num1 else (col1, col2)
            fig = px.box(self.df, x=x, y=y, points="all", title=f'{y} distributed by {x}')
        else:
            df_counts = self.df.groupby([col1, col2]).size().reset_index(name='count')
            fig = px.bar(df_counts, x=col1, y='count', color=col2, barmode='group', title=f'{col1} grouped by {col2}')
        fig.show()

    def plot_all_associations_heatmap(self):
        """Generates an association heatmap for numeric data."""
        num_df = self.df.select_dtypes(include=['float64', 'int64'])
        if not num_df.empty:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=True, title="Data Associations Heatmap")
            fig.show()
        else:
            print("Not enough numeric data for a heatmap.")
