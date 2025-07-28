import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
import openai
from typing import Dict, List, Tuple, Any
import warnings
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
import phonenumbers
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🧽 Clean My Spreadsheet Pro",
    page_icon="🧽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedSpreadsheetCleaner:
    def __init__(self):
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_summary = []
        self.data_quality_score = {}
        self.outliers_detected = {}
        self.suggestions = []
        
    def validate_email(self, email: str) -> str:
        """Validate and clean email addresses"""
        if pd.isna(email):
            return email
        
        email = str(email).strip().lower()
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, email):
            return email
        else:
            return np.nan  # Mark invalid emails as missing
    
    def standardize_phone(self, phone: str) -> str:
        """Standardize phone numbers"""
        if pd.isna(phone):
            return phone
        
        try:
            # Clean the phone number string
            phone_str = re.sub(r'[^\d+]', '', str(phone))
            
            # Try to parse and format
            if not phone_str.startswith('+'):
                phone_str = '+1' + phone_str  # Assume US if no country code
            
            parsed = phonenumbers.parse(phone_str, None)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            else:
                return np.nan
        except:
            return np.nan
    
    def clean_currency(self, value: str) -> float:
        """Clean currency values"""
        if pd.isna(value):
            return value
        
        # Remove currency symbols and convert to float
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        try:
            return float(cleaned)
        except:
            return np.nan
    
    def detect_and_clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and clean data types"""
        type_corrections = 0
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Email validation
            if 'email' in col_lower:
                original_valid = df[col].notna().sum()
                df[col] = df[col].apply(self.validate_email)
                new_valid = df[col].notna().sum()
                invalid_emails = original_valid - new_valid
                if invalid_emails > 0:
                    self.cleaning_summary.append(f"✅ Validated emails in '{col}' - flagged {invalid_emails} invalid addresses")
                type_corrections += 1
            
            # Phone number standardization
            elif any(keyword in col_lower for keyword in ['phone', 'tel', 'mobile', 'cell']):
                try:
                    original_valid = df[col].notna().sum()
                    df[col] = df[col].apply(self.standardize_phone)
                    new_valid = df[col].notna().sum()
                    invalid_phones = original_valid - new_valid
                    if invalid_phones > 0:
                        self.cleaning_summary.append(f"✅ Standardized phones in '{col}' - flagged {invalid_phones} invalid numbers")
                    type_corrections += 1
                except:
                    pass
            
            # Currency cleaning
            elif any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'revenue', 'salary', 'wage']):
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(self.clean_currency)
                    self.cleaning_summary.append(f"✅ Cleaned currency format in '{col}'")
                    type_corrections += 1
            
            # Numeric conversion
            elif df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                if not numeric_converted.isna().all():
                    non_null_original = df[col].notna().sum()
                    non_null_converted = numeric_converted.notna().sum()
                    
                    # Only convert if we don't lose too much data
                    if non_null_converted >= non_null_original * 0.8:
                        df[col] = numeric_converted
                        type_corrections += 1
        
        if type_corrections > 0:
            self.cleaning_summary.append(f"✅ Applied smart data type corrections to {type_corrections} columns")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, List]:
        """Detect statistical outliers using IQR method"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Only process if we have data
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_indices = df[outlier_mask].index.tolist()
                    
                    if outlier_indices:
                        outliers[col] = {
                            'indices': outlier_indices,
                            'count': len(outlier_indices),
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
        
        self.outliers_detected = outliers
        if outliers:
            total_outliers = sum(data['count'] for data in outliers.values())
            self.cleaning_summary.append(f"🔍 Detected {total_outliers} potential outliers across {len(outliers)} columns")
        
        return outliers
    
    def suggest_column_merges(self, df: pd.DataFrame) -> List[Dict]:
        """Suggest columns that might be duplicates or should be merged"""
        suggestions = []
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                similarity = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
                if similarity > 0.8:  # 80% similar
                    suggestions.append({
                        'columns': [col1, col2],
                        'similarity': f"{similarity:.2%}",
                        'action': 'Consider merging similar columns'
                    })
        
        # Check for potential name/address splitting
        name_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['first', 'last', 'fname', 'lname'])]
        if len(name_cols) >= 2:
            suggestions.append({
                'columns': name_cols,
                'similarity': 'N/A',
                'action': 'Consider combining name columns'
            })
        
        return suggestions
    
    def calculate_quality_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive data quality score"""
        total_cells = len(df) * len(df.columns)
        
        scores = {
            'completeness': 100 - (df.isnull().sum().sum() / total_cells * 100),
            'uniqueness': (df.nunique().sum() / total_cells * 100),
            'consistency': self.check_format_consistency(df),
            'validity': self.check_data_validity(df)
        }
        
        overall_score = sum(scores.values()) / len(scores)
        scores['overall'] = overall_score
        
        return scores
    
    def check_format_consistency(self, df: pd.DataFrame) -> float:
        """Check format consistency within columns"""
        consistency_scores = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Check format patterns
                    patterns = [len(str(val)) for val in non_null_values.head(100)]
                    if patterns:
                        pattern_consistency = 1 - (np.std(patterns) / np.mean(patterns)) if np.mean(patterns) > 0 else 1
                        consistency_scores.append(max(0, min(1, pattern_consistency)) * 100)
                    else:
                        consistency_scores.append(100)
                else:
                    consistency_scores.append(100)
            else:
                consistency_scores.append(100)  # Numeric data is inherently consistent
        
        return np.mean(consistency_scores) if consistency_scores else 100
    
    def check_data_validity(self, df: pd.DataFrame) -> float:
        """Check data validity based on data types and patterns"""
        validity_scores = []
        
        for col in df.columns:
            col_lower = col.lower()
            valid_count = 0
            total_count = df[col].notna().sum()
            
            if total_count == 0:
                validity_scores.append(100)
                continue
            
            if 'email' in col_lower:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_count = df[col].dropna().apply(lambda x: bool(re.match(email_pattern, str(x)))).sum()
            elif any(keyword in col_lower for keyword in ['phone', 'tel']):
                # Simple phone validation
                phone_pattern = r'[\+]?[\d\s\-\(\)]{7,}'
                valid_count = df[col].dropna().apply(lambda x: bool(re.match(phone_pattern, str(x)))).sum()
            else:
                valid_count = total_count  # Assume valid if no specific pattern
            
            validity_score = (valid_count / total_count * 100) if total_count > 0 else 100
            validity_scores.append(validity_score)
        
        return np.mean(validity_scores) if validity_scores else 100
    
    def clean_column_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column headers with comprehensive handling"""
        new_columns = []
        for col in df.columns:
            # Convert to string and clean
            clean_col = str(col).strip()
            
            # Handle completely empty or numeric column names
            if not clean_col or clean_col.isdigit():
                clean_col = f"Column_{len(new_columns) + 1}"
            
            # Remove special characters except spaces, underscores, and alphanumeric
            clean_col = re.sub(r'[^\w\s]', '', clean_col)
            
            # Replace underscores and multiple spaces with single spaces
            clean_col = re.sub(r'[_\s]+', ' ', clean_col)
            
            # Remove leading/trailing spaces
            clean_col = clean_col.strip()
            
            # Handle empty result after cleaning
            if not clean_col:
                clean_col = f"Column_{len(new_columns) + 1}"
            
            # Apply title case
            clean_col = clean_col.title()
            
            # Handle common abbreviations with comprehensive word boundary matching
            # This handles cases like "qty_amt", "tel_num", "api_url", etc.
            abbreviation_map = {
                # Technical terms
                r'\bId\b': 'ID',
                r'\bIds\b': 'IDs', 
                r'\bUrl\b': 'URL',
                r'\bUrls\b': 'URLs',
                r'\bApi\b': 'API',
                r'\bApis\b': 'APIs',
                r'\bHttp\b': 'HTTP',
                r'\bHttps\b': 'HTTPS',
                r'\bXml\b': 'XML',
                r'\bJson\b': 'JSON',
                r'\bCsv\b': 'CSV',
                r'\bPdf\b': 'PDF',
                r'\bSql\b': 'SQL',
                
                # Business terms
                r'\bRev\b': 'Revenue',
                r'\bQty\b': 'Quantity',
                r'\bAmt\b': 'Amount',
                r'\bNum\b': 'Number',
                r'\bTel\b': 'Telephone',
                r'\bEmail\b': 'Email',
                r'\bAddr\b': 'Address',
                r'\bSt\b': 'Street',
                r'\bAve\b': 'Avenue',
                r'\bBlvd\b': 'Boulevard',
                r'\bLtd\b': 'Limited',
                r'\bInc\b': 'Incorporated',
                r'\bCorp\b': 'Corporation',
                
                # Date/Time terms
                r'\bDt\b': 'Date',
                r'\bTm\b': 'Time',
                r'\bYr\b': 'Year',
                r'\bMth\b': 'Month',
                r'\bWk\b': 'Week',
                r'\bHr\b': 'Hour',
                r'\bMin\b': 'Minute',
                r'\bSec\b': 'Second',
                
                # Quarter terms
                r'\bQ1\b': 'Q1',
                r'\bQ2\b': 'Q2', 
                r'\bQ3\b': 'Q3',
                r'\bQ4\b': 'Q4',
                
                # Common business abbreviations
                r'\bCust\b': 'Customer',
                r'\bProd\b': 'Product',
                r'\bDesc\b': 'Description',
                r'\bCat\b': 'Category',
                r'\bDept\b': 'Department',
                r'\bMgr\b': 'Manager',
                r'\bAcct\b': 'Account',
                r'\bTxn\b': 'Transaction',
                r'\bOrg\b': 'Organization',
                r'\bRef\b': 'Reference',
                r'\bStat\b': 'Status',
                r'\bType\b': 'Type',
                r'\bLoc\b': 'Location',
                r'\bPos\b': 'Position',
                
                # Measurement terms
                r'\bPct\b': 'Percent',
                r'\bTemp\b': 'Temperature',
                r'\bWt\b': 'Weight',
                r'\bHt\b': 'Height',
                r'\bLen\b': 'Length',
                r'\bWdth\b': 'Width',
                r'\bVol\b': 'Volume'
            }
            
            # Apply abbreviation replacements
            for pattern, replacement in abbreviation_map.items():
                clean_col = re.sub(pattern, replacement, clean_col, flags=re.IGNORECASE)
            
            # Ensure proper title case after replacements
            clean_col = clean_col.title()
            
            # Handle duplicate column names
            original_clean = clean_col
            counter = 1
            while clean_col in new_columns:
                clean_col = f"{original_clean} {counter}"
                counter += 1
            
            new_columns.append(clean_col)
        
        df.columns = new_columns
        self.cleaning_summary.append(f"✅ Cleaned {len(df.columns)} column headers with {len([c for c in new_columns if c != df.columns[new_columns.index(c)]])} improvements")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        if duplicates_removed > 0:
            self.cleaning_summary.append(f"✅ Removed {duplicates_removed} duplicate rows")
        else:
            self.cleaning_summary.append("✅ No duplicate rows found")
        
        return df
    
    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows"""
        initial_rows = len(df)
        df = df.dropna(how='all')
        empty_rows_removed = initial_rows - len(df)
        
        if empty_rows_removed > 0:
            self.cleaning_summary.append(f"✅ Removed {empty_rows_removed} completely empty rows")
        else:
            self.cleaning_summary.append("✅ No completely empty rows found")
        
        return df
    
    def standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to parse and standardize date columns"""
        date_columns_found = 0
        
        for col in df.columns:
            # Check if column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'modified']):
                try:
                    # Try to parse as datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    # Format as standard date string
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
                    date_columns_found += 1
                except:
                    continue
        
        if date_columns_found > 0:
            self.cleaning_summary.append(f"✅ Standardized {date_columns_found} date columns")
        else:
            self.cleaning_summary.append("✅ No date columns detected")
        
        return df
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality and flag issues"""
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'empty_cells_percent': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'suspicious_columns': [],
            'columns_with_missing_data': {}
        }
        
        # Find columns with high percentage of missing data
        for col in df.columns:
            missing_percent = (df[col].isnull().sum() / len(df)) * 100
            if missing_percent > 50:
                analysis['suspicious_columns'].append(col)
            if missing_percent > 0:
                analysis['columns_with_missing_data'][col] = missing_percent
        
        return analysis
    
    def clean_spreadsheet(self, df: pd.DataFrame, advanced_mode: bool = False) -> pd.DataFrame:
        """Main cleaning function with optional advanced features"""
        self.original_data = df.copy()
        self.cleaning_summary = []
        
        # Apply basic cleaning steps
        df = self.clean_column_headers(df)
        df = self.remove_duplicates(df)
        df = self.remove_empty_rows(df)
        df = self.standardize_dates(df)
        
        # Apply advanced cleaning if enabled
        if advanced_mode:
            df = self.detect_and_clean_data_types(df)
            self.detect_outliers(df)
            self.suggestions = self.suggest_column_merges(df)
            self.data_quality_score = self.calculate_quality_score(df)
        
        self.cleaned_data = df
        return df

def create_data_quality_chart(scores: Dict[str, float]) -> go.Figure:
    """Create a radar chart for data quality scores"""
    categories = ['Completeness', 'Uniqueness', 'Consistency', 'Validity']
    values = [scores.get('completeness', 0), scores.get('uniqueness', 0), 
              scores.get('consistency', 0), scores.get('validity', 0)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Data Quality Score',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Data Quality Assessment",
        height=400
    )
    
    return fig

def create_missing_data_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing missing data by column"""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df) * 100).round(2)
    
    fig = px.bar(
        x=missing_percent.index,
        y=missing_percent.values,
        title="Missing Data by Column",
        labels={'x': 'Columns', 'y': 'Missing Data (%)'},
        color=missing_percent.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def generate_ai_summary(df: pd.DataFrame, analysis: Dict[str, Any], api_key: str = None, 
                       tone: str = "Professional", include_suggestions: bool = True) -> str:
    """Generate AI summary of the spreadsheet with customizable tone and suggestions"""
    try:
        if not api_key:
            # Enhanced fallback summary without AI
            summary = f"📊 **Data Summary**: {analysis['total_rows']} records across {analysis['total_columns']} columns. "
            summary += f"Data completeness: {100 - analysis['empty_cells_percent']:.1f}%. "
            
            if analysis['suspicious_columns']:
                summary += f"⚠️ {len(analysis['suspicious_columns'])} columns have >50% missing data. "
            
            # Add insights about data types
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            summary += f"Contains {numeric_cols} numeric and {text_cols} text columns. "
            
            if include_suggestions:
                summary += "💡 Consider reviewing columns with high missing data and validating data types for better analysis."
            
            return summary
        
        # Define tone-specific instructions
        tone_instructions = {
            "Professional": "Use formal business language, focus on data quality metrics and actionable insights.",
            "Friendly": "Use conversational, approachable language that's easy to understand. Be encouraging and helpful.",
            "Executive": "Use concise, high-level language focused on business impact and strategic insights. Be direct and authoritative.",
            "Technical": "Use precise technical terminology, include statistical details and data science perspectives."
        }
        
        tone_instruction = tone_instructions.get(tone, tone_instructions["Professional"])
        
        # AI-powered summary with tone customization
        openai.api_key = api_key
        
        base_prompt = f"""
        Analyze this spreadsheet data and provide a summary in a {tone.lower()} tone.
        
        {tone_instruction}
        
        Data Summary:
        - Total rows: {analysis['total_rows']}
        - Total columns: {analysis['total_columns']}
        - Data completeness: {100 - analysis['empty_cells_percent']:.1f}%
        - Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}
        - Suspicious columns (>50% missing): {analysis['suspicious_columns']}
        
        Sample data (first few rows):
        {df.head(3).to_string()}
        
        Provide a 2-3 sentence summary focusing on what this data contains and data quality.
        """
        
        if include_suggestions:
            base_prompt += f"""
            
            Additionally, provide 1-2 specific, actionable suggestions for improving this dataset's quality or usefulness.
            """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": base_prompt}],
            max_tokens=200,
            temperature=0.7 if tone == "Friendly" else 0.5
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        # Enhanced fallback if AI fails
        fallback_summary = f"📊 **Data Summary**: {analysis['total_rows']} records with {analysis['total_columns']} columns. Data completeness: {100 - analysis['empty_cells_percent']:.1f}%."
        
        if include_suggestions:
            fallback_summary += " 💡 Consider enabling advanced cleaning mode for deeper insights."
        
        return fallback_summary

def load_file(uploaded_file) -> pd.DataFrame:
    """Load uploaded file into pandas DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_pdf_report(df: pd.DataFrame, analysis: Dict[str, Any], cleaning_summary: List[str], 
                     ai_summary: str, quality_scores: Dict = None, outliers: Dict = None, 
                     suggestions: List = None, include_sections: Dict = None) -> bytes:
    """Create a comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.HexColor('#667eea')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#764ba2')
    )
    
    # Build story
    story = []
    
    # Title
    if include_sections.get('title', True):
        story.append(Paragraph("🧽 Spreadsheet Cleaning Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report metadata
        metadata = [
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            f"Total Records: {analysis['total_rows']:,}",
            f"Total Columns: {analysis['total_columns']}",
            f"Data Completeness: {100 - analysis['empty_cells_percent']:.1f}%"
        ]
        
        for item in metadata:
            story.append(Paragraph(item, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Executive Summary
    if include_sections.get('summary', True):
        story.append(Paragraph("📊 Executive Summary", heading_style))
        story.append(Paragraph(ai_summary, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Data Quality Scores
    if quality_scores and include_sections.get('quality', True):
        story.append(Paragraph("📈 Data Quality Assessment", heading_style))
        
        quality_data = [
            ['Metric', 'Score', 'Status'],
            ['Overall Quality', f"{quality_scores.get('overall', 0):.1f}/100", 
             'Excellent' if quality_scores.get('overall', 0) >= 80 else 'Good' if quality_scores.get('overall', 0) >= 60 else 'Needs Improvement'],
            ['Completeness', f"{quality_scores.get('completeness', 0):.1f}%", '✓'],
            ['Uniqueness', f"{quality_scores.get('uniqueness', 0):.1f}%", '✓'],
            ['Consistency', f"{quality_scores.get('consistency', 0):.1f}%", '✓'],
            ['Validity', f"{quality_scores.get('validity', 0):.1f}%", '✓']
        ]
        
        quality_table = Table(quality_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(quality_table)
        story.append(Spacer(1, 20))
    
    # Cleaning Actions
    if include_sections.get('actions', True):
        story.append(Paragraph("🧽 Cleaning Actions Performed", heading_style))
        for action in cleaning_summary:
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Data Overview
    if include_sections.get('overview', True):
        story.append(Paragraph("📋 Data Overview", heading_style))
        
        overview_data = [
            ['Column Name', 'Data Type', 'Non-Null Count', 'Missing %']
        ]
        
        for col in df.columns[:15]:  # Limit to first 15 columns for space
            missing_pct = (df[col].isnull().sum() / len(df) * 100)
            overview_data.append([
                col,
                str(df[col].dtype),
                f"{df[col].notna().sum():,}",
                f"{missing_pct:.1f}%"
            ])
        
        if len(df.columns) > 15:
            overview_data.append(['...', '...', '...', '...'])
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 1*inch, 1*inch, 0.8*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 20))
    
    # Outliers
    if outliers and include_sections.get('outliers', True):
        story.append(Paragraph("🔍 Outliers Detected", heading_style))
        if outliers:
            for col, data in outliers.items():
                story.append(Paragraph(f"• <b>{col}</b>: {data['count']} outliers detected", styles['Normal']))
        else:
            story.append(Paragraph("No significant outliers detected.", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Recommendations
    if suggestions and include_sections.get('recommendations', True):
        story.append(Paragraph("💡 Recommendations", heading_style))
        if suggestions:
            for suggestion in suggestions:
                story.append(Paragraph(f"• {suggestion['action']}: {', '.join(suggestion['columns'])}", styles['Normal']))
        else:
            story.append(Paragraph("No specific recommendations at this time.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def create_download_link(df: pd.DataFrame, filename: str) -> bytes:
    """Create downloadable Excel file"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    
    return output.getvalue()
    """Create downloadable Excel file"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    
    return output.getvalue()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧽 Clean My Spreadsheet Pro</h1>
        <p>Advanced data cleaning with AI-powered insights and quality scoring!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = AdvancedSpreadsheetCleaner()
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    
    # Sidebar for advanced options
    with st.sidebar:
        st.markdown("### ⚙️ Advanced Options")
        advanced_mode = st.checkbox("Enable Advanced Cleaning", value=True, 
                                   help="Includes data type validation, outlier detection, and quality scoring")
        
        show_charts = st.checkbox("Show Data Quality Charts", value=True)
        
        st.markdown("### 🤖 AI Summary Settings")
        summary_tone = st.selectbox(
            "Summary Tone",
            ["Professional", "Friendly", "Executive", "Technical"],
            help="Choose the tone for AI-generated insights"
        )
        
        include_suggestions = st.checkbox(
            "Include Improvement Suggestions", 
            value=True,
            help="Add actionable recommendations to the summary"
        )
        
        st.markdown("### 📥 Download Settings")
        custom_filename = st.text_input(
            "Custom Filename (optional)",
            placeholder="my_clean_data",
            help="Leave blank to use original filename"
        )
        
        # Report sections to include
        st.markdown("**PDF Report Sections:**")
        include_title = st.checkbox("Title & Metadata", value=True)
        include_summary = st.checkbox("Executive Summary", value=True)
        include_quality = st.checkbox("Quality Scores", value=True)
        include_actions = st.checkbox("Cleaning Actions", value=True)
        include_overview = st.checkbox("Data Overview", value=True)
        include_outliers = st.checkbox("Outliers Analysis", value=True)
        include_recommendations = st.checkbox("Recommendations", value=True)
        
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="For AI-powered insights and personalized summaries"
        )
    
    # Step 1: File Upload
    st.markdown("### 📁 Step 1: Upload Your Spreadsheet")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: .csv, .xlsx, .xls (max 200MB)"
    )
    
    if uploaded_file is not None:
        # Load and preview original data
        df = load_file(uploaded_file)
        
        if df is not None:
            st.markdown(f"""
            <div class="feature-box">
                <h4>📋 Original File Preview</h4>
                <p><strong>File:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {len(df)} rows × {len(df.columns)} columns</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show preview of original data
            st.dataframe(df.head(10), use_container_width=True)
            
            # Step 2: Clean the data
            st.markdown("### 🧽 Step 2: Clean Your Data")
            
            if st.button("🚀 Clean My Spreadsheet", type="primary"):
                with st.spinner("Cleaning your spreadsheet..."):
                    # Clean the data
                    cleaned_df = st.session_state.cleaner.clean_spreadsheet(df, advanced_mode)
                    st.session_state.file_processed = True
            
            # Step 3: Show results if processed
            if st.session_state.file_processed and st.session_state.cleaner.cleaned_data is not None:
                cleaned_df = st.session_state.cleaner.cleaned_data
                
                st.markdown("### ✨ Step 3: Review Your Cleaned Data")
                
                # Show cleaning summary
                st.markdown("""
                <div class="success-box">
                    <h4>🎉 Cleaning Complete!</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display cleaning actions
                for action in st.session_state.cleaner.cleaning_summary:
                    st.markdown(f"- {action}")
                
                # Data analysis
                analysis = st.session_state.cleaner.analyze_data_quality(cleaned_df)
                
                # Show advanced features if enabled
                if advanced_mode:
                    # Data Quality Score
                    if st.session_state.cleaner.data_quality_score:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("### 📊 Data Quality Score")
                            scores = st.session_state.cleaner.data_quality_score
                            
                            # Overall score with color coding
                            overall_score = scores.get('overall', 0)
                            score_color = 'green' if overall_score >= 80 else 'orange' if overall_score >= 60 else 'red'
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; border: 2px solid {score_color};">
                                <h2 style="color: {score_color}; margin: 0;">{overall_score:.1f}/100</h2>
                                <p style="margin: 0;">Overall Quality Score</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Individual scores
                            for metric, score in scores.items():
                                if metric != 'overall':
                                    st.metric(metric.title(), f"{score:.1f}%")
                        
                        with col2:
                            if show_charts:
                                st.plotly_chart(create_data_quality_chart(scores), use_container_width=True)
                    
                    # Outliers Detection
                    if st.session_state.cleaner.outliers_detected:
                        st.markdown("### 🔍 Outliers Detected")
                        outliers = st.session_state.cleaner.outliers_detected
                        
                        for col, data in outliers.items():
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>Column: {col}</strong><br>
                                Found {data['count']} outliers (values outside {data['lower_bound']:.2f} - {data['upper_bound']:.2f})
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Column Suggestions
                    if st.session_state.cleaner.suggestions:
                        st.markdown("### 💡 Improvement Suggestions")
                        for suggestion in st.session_state.cleaner.suggestions:
                            st.info(f"**{suggestion['action']}**: {', '.join(suggestion['columns'])} (Similarity: {suggestion['similarity']})")
                
                # Charts
                if show_charts:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(create_missing_data_chart(cleaned_df), use_container_width=True)
                    
                    with col2:
                        # Data type distribution
                        data_types = cleaned_df.dtypes.value_counts()
                        # Convert data types to string to avoid JSON serialization issues
                        type_names = [str(dtype) for dtype in data_types.index]
                        type_counts = data_types.values.tolist()
                        
                        fig = px.pie(values=type_counts, names=type_names, 
                                   title="Data Types Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                # AI Summary
                st.markdown("### 🤖 Data Insights")
                ai_summary = generate_ai_summary(cleaned_df, analysis, api_key, summary_tone, include_suggestions)
                st.markdown(f"""
                <div class="feature-box">
                    {ai_summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Show cleaned data preview
                st.markdown("### 📊 Cleaned Data Preview")
                st.dataframe(cleaned_df.head(10), use_container_width=True)
                
                # Data quality metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", analysis['total_rows'])
                with col2:
                    st.metric("Total Columns", analysis['total_columns'])
                with col3:
                    st.metric("Data Completeness", f"{100 - analysis['empty_cells_percent']:.1f}%")
                with col4:
                    quality_issues = len(analysis['suspicious_columns'])
                    if advanced_mode and st.session_state.cleaner.outliers_detected:
                        quality_issues += len(st.session_state.cleaner.outliers_detected)
                    st.metric("Quality Issues", quality_issues)
                
                # Step 4: Download
                st.markdown("### 📥 Step 4: Download Your Clean Data")
                
                # Determine filename
                base_filename = custom_filename if custom_filename.strip() else uploaded_file.name.split('.')[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download cleaned Excel file
                    excel_data = create_download_link(cleaned_df, "cleaned_data.xlsx")
                    st.download_button(
                        label="📊 Download Excel File",
                        data=excel_data,
                        file_name=f"cleaned_{base_filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # Download comprehensive summary report
                    report_sections = [
                        "ADVANCED SPREADSHEET CLEANING REPORT",
                        "=" * 50,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Original File: {uploaded_file.name}",
                        f"Processing Mode: {'Advanced' if advanced_mode else 'Standard'}",
                        "",
                        "CLEANING ACTIONS PERFORMED:",
                        "-" * 30
                    ]
                    
                    report_sections.extend([f"• {action}" for action in st.session_state.cleaner.cleaning_summary])
                    
                    report_sections.extend([
                        "",
                        "DATA ANALYSIS:",
                        "-" * 15,
                        ai_summary,
                        "",
                        "QUALITY METRICS:",
                        "-" * 16,
                        f"• Total Rows: {analysis['total_rows']}",
                        f"• Total Columns: {analysis['total_columns']}",
                        f"• Data Completeness: {100 - analysis['empty_cells_percent']:.1f}%",
                        f"• Columns with Missing Data: {len(analysis['columns_with_missing_data'])}"
                    ])
                    
                    if advanced_mode and st.session_state.cleaner.data_quality_score:
                        scores = st.session_state.cleaner.data_quality_score
                        report_sections.extend([
                            "",
                            "ADVANCED QUALITY SCORES:",
                            "-" * 25,
                            f"• Overall Score: {scores.get('overall', 0):.1f}/100",
                            f"• Completeness: {scores.get('completeness', 0):.1f}%",
                            f"• Uniqueness: {scores.get('uniqueness', 0):.1f}%",
                            f"• Consistency: {scores.get('consistency', 0):.1f}%",
                            f"• Validity: {scores.get('validity', 0):.1f}%"
                        ])
                    
                    if st.session_state.cleaner.outliers_detected:
                        report_sections.extend([
                            "",
                            "OUTLIERS DETECTED:",
                            "-" * 18
                        ])
                        for col, data in st.session_state.cleaner.outliers_detected.items():
                            report_sections.append(f"• {col}: {data['count']} outliers detected")
                    
                    if st.session_state.cleaner.suggestions:
                        report_sections.extend([
                            "",
                            "IMPROVEMENT SUGGESTIONS:",
                            "-" * 25
                        ])
                        for suggestion in st.session_state.cleaner.suggestions:
                            report_sections.append(f"• {suggestion['action']}: {', '.join(suggestion['columns'])}")
                    
                    summary_text = "\n".join(report_sections)
                    
                    st.download_button(
                        label="📄 Download Text Report",
                        data=summary_text,
                        file_name=f"report_{base_filename}.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    # Download PDF Report
                    if advanced_mode:
                        try:
                            include_sections = {
                                'title': include_title,
                                'summary': include_summary,
                                'quality': include_quality,
                                'actions': include_actions,
                                'overview': include_overview,
                                'outliers': include_outliers,
                                'recommendations': include_recommendations
                            }
                            
                            pdf_data = create_pdf_report(
                                cleaned_df, 
                                analysis, 
                                st.session_state.cleaner.cleaning_summary,
                                ai_summary,
                                st.session_state.cleaner.data_quality_score,
                                st.session_state.cleaner.outliers_detected,
                                st.session_state.cleaner.suggestions,
                                include_sections
                            )
                            
                            st.download_button(
                                label="📋 Download PDF Report",
                                data=pdf_data,
                                file_name=f"professional_report_{base_filename}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error("PDF generation requires additional setup. Download text report instead.")
                    else:
                        st.info("📋 PDF reports available in Advanced Mode")
                
                # Additional export options
                st.markdown("### 🎯 Additional Export Formats")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV export
                    csv_data = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv_data,
                        file_name=f"cleaned_{base_filename}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # JSON export
                    json_data = cleaned_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="🔗 Download as JSON",
                        data=json_data,
                        file_name=f"cleaned_{base_filename}.json",
                        mime="application/json"
                    )
                
                with col3:
                    # Data dictionary export
                    data_dict_sections = [
                        "DATA DICTIONARY",
                        "=" * 15,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Source: {uploaded_file.name}",
                        "",
                        "COLUMN INFORMATION:",
                        "-" * 20
                    ]
                    
                    for col in cleaned_df.columns:
                        col_info = [
                            f"Column: {col}",
                            f"  Data Type: {cleaned_df[col].dtype}",
                            f"  Non-null Count: {cleaned_df[col].notna().sum()}",
                            f"  Missing Count: {cleaned_df[col].isnull().sum()}",
                            f"  Unique Values: {cleaned_df[col].nunique()}"
                        ]
                        
                        if cleaned_df[col].dtype in ['int64', 'float64']:
                            col_info.extend([
                                f"  Min Value: {cleaned_df[col].min()}",
                                f"  Max Value: {cleaned_df[col].max()}",
                                f"  Mean: {cleaned_df[col].mean():.2f}"
                            ])
                        
                        data_dict_sections.extend(col_info)
                        data_dict_sections.append("")
                    
                    data_dict_text = "\n".join(data_dict_sections)
                    
                    st.download_button(
                        label="📖 Download Data Dictionary",
                        data=data_dict_text,
                        file_name=f"dictionary_{base_filename}.txt",
                        mime="text/plain"
                    )
    
    # Footer with enhanced features
    st.markdown("---")
    st.markdown("### 🚀 What This Pro Version Does")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🧽 Smart Cleaning**
        - Clean column headers
        - Remove duplicates
        - Delete empty rows
        - Standardize dates
        - Data type validation
        """)
    
    with col2:
        st.markdown("""
        **🔍 Advanced Analysis**
        - Outlier detection
        - Quality scoring
        - Format consistency
        - Data validity checks
        - Column suggestions
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI Features**
        - Intelligent insights
        - Pattern recognition
        - Quality assessment
        - Smart recommendations
        - Context analysis
        """)
    
    with col4:
        st.markdown("""
        **📊 Rich Exports**
        - Excel with formatting
        - Detailed reports
        - Multiple formats
        - Data dictionary
        - Quality charts
        """)
    
    # Pro tips section
    st.markdown("### 💡 Pro Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **🎯 For Best Results:**
        - Enable Advanced Cleaning for comprehensive analysis
        - Use meaningful column names in your source data
        - Include an OpenAI API key for AI-powered insights
        - Review outliers before making decisions
        """)
    
    with tips_col2:
        st.markdown("""
        **📈 Quality Scoring:**
        - **80-100**: Excellent data quality
        - **60-79**: Good with minor issues
        - **40-59**: Fair, needs attention
        - **Below 40**: Poor, requires significant cleaning
        """)
    
    # Feedback section
    st.markdown("### 📧 Feedback & Support")
    
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    
    with feedback_col1:
        st.markdown("""
        **🐛 Found a Bug?**
        Report issues and we'll fix them quickly.
        Your feedback helps us improve!
        """)
    
    with feedback_col2:
        st.markdown("""
        **💡 Feature Request?**
        Suggest new cleaning features or 
        improvements you'd like to see.
        """)
    
    with feedback_col3:
        st.markdown("""
        **🚀 Enterprise?**
        Need custom features, API access,
        or white-label solutions?
        """)

if __name__ == "__main__":
    main()