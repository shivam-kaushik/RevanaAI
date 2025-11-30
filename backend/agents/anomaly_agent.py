import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnomalyAgent:
    def __init__(self, contamination=0.05):
        """Initialize the Anomaly Detection Agent
        
            contamination: Expected proportion of outliers in the data (0.0 to 0.5)
                Default 0.05 means we expect ~5% of data to be anomalies
        """
        self.contamination = contamination
        logger.info("Anomaly detection agent initialized")
    
    def _auto_detect_date_column(self, data):
        """Intelligently detect the date/time column"""
        date_keywords = ['date', 'time', 'timestamp', 'invoicedate', 'orderdate', 'transactiondate']
        
        for col in data.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if any(keyword in col_lower for keyword in date_keywords):
                return col
        
        # Fallback: check data types
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                return col
        
        return None
    
    def _auto_detect_value_column(self, data):
        """Intelligently detect the numeric value column for anomaly detection"""
        # Priority order for value columns
        value_keywords = [
            'total_amount', 'totalamount', 'amount', 'total',
            'quantity', 'qty', 
            'unitprice', 'price', 
            'revenue', 'sales', 'value'
        ]
        
        for keyword in value_keywords:
            for col in data.columns:
                col_lower = col.lower().replace('_', '').replace(' ', '')
                if keyword == col_lower:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        return col
        
        # Fallback: find first numeric column
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                return col
        
        return None
    
    def _auto_detect_category_column(self, data):
        """Intelligently detect the category/grouping column"""
        category_keywords = [
            'product_category', 'productcategory', 'category',
            'product', 'description', 'item', 'productitem',
            'type', 'class', 'group', 'segment'
        ]
        
        for keyword in category_keywords:
            for col in data.columns:
                col_lower = col.lower().replace('_', '').replace(' ', '')
                if keyword in col_lower:
                    return col
        
        return None
    
    def detect_anomalies(self, data, time_column=None, value_column=None):
        """Detect anomalies using IsolationForest or Z-score method
            data (pd.DataFrame): Input dataframe with time and value columns
            time_column: Optional - will auto-detect if not provided
            value_column: Optional - will auto-detect if not provided
            Detection results including anomalies, narratives, plot, and statistics
        """
        try:
            logger.info(f"Detecting anomalies in {len(data)} data points")
            
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Auto-detect columns if not provided
            if time_column is None:
                time_column = self._auto_detect_date_column(data)
                if time_column:
                    logger.info(f"📅 Auto-detected date column: {time_column}")
                else:
                    return {
                        'success': False,
                        'error': "Could not find a date/time column in the data"
                    }
            
            if value_column is None:
                value_column = self._auto_detect_value_column(data)
                if value_column:
                    logger.info(f"💰 Auto-detected value column: {value_column}")
                else:
                    return {
                        'success': False,
                        'error': "Could not find a numeric value column in the data"
                    }
            
            # Validate required columns exist
            if time_column not in data.columns or value_column not in data.columns:
                return {
                    'success': False,
                    'error': f"Required columns {time_column} and/or {value_column} not found"
                }
            
            # Prepare datetime column
            data = data.copy()
            data[time_column] = self._prepare_datetime_column(data[time_column])
            
            # Prepare value column
            data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
            
            # Remove rows with NaN values
            data = data.dropna(subset=[time_column, value_column])
            
            if len(data) < 3:
                return {
                    'success': False,
                    'error': 'Not enough data points for anomaly detection (minimum 3 required)'
                }
            
            # Sort by time
            data = data.sort_values(by=time_column).reset_index(drop=True)
            
            # Detect anomalies
            if len(data) < 5:
                is_anomaly, method = self._detect_zscore(data[value_column])
            else:
                is_anomaly, method = self._detect_isolation_forest(data[value_column])
            
            # Separate anomalies from normal data
            anomalies = data[is_anomaly].copy()
            normal_data = data[~is_anomaly].copy()
            
            if anomalies.empty:
                # Create visualization even when no anomalies found
                fig = self._generate_plot(data, pd.DataFrame(), time_column, value_column)
                return {
                    'success': True,
                    'method': method,
                    'message': 'No anomalies detected',
                    'plot': fig,
                    'statistics': {
                        'total_points': len(data),
                        'anomalies_found': 0,
                        'anomaly_rate': 0
                    }
                }
            
            # Calculate deviations and generate narratives
            baseline = normal_data[value_column].mean()
            anomalies['deviation'] = ((anomalies[value_column] - baseline) / baseline * 100)
            narratives = self._generate_narratives(anomalies, time_column)
            
            # Create visualization
            fig = self._generate_plot(data, anomalies, time_column, value_column)
            
            return {
                'success': True,
                'method': method,
                'anomalies': anomalies.to_dict('records'),
                'narratives': narratives,
                'plot': fig,
                'statistics': {
                    'total_points': len(data),
                    'anomalies_found': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(data),
                    'baseline': round(baseline, 2)
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Anomaly detection error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def detect_anomalies_by_category(self, data, time_column=None, value_column=None, category_column=None):
        """Detect anomalies per category and generate a combined visualization.
            time_column: Optional - will auto-detect if not provided
            value_column: Optional - will auto-detect if not provided
            category_column: Optional - will auto-detect if not provided
        """
        try:
            # Basic validation
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Auto-detect columns if not provided
            if time_column is None:
                time_column = self._auto_detect_date_column(data)
                if time_column:
                    logger.info(f"📅 Auto-detected date column: {time_column}")
                else:
                    return {
                        'success': False,
                        'error': "Could not find a date/time column in the data"
                    }
            
            if value_column is None:
                value_column = self._auto_detect_value_column(data)
                if value_column:
                    logger.info(f"💰 Auto-detected value column: {value_column}")
                else:
                    return {
                        'success': False,
                        'error': "Could not find a numeric value column in the data"
                    }
            
            if category_column is None:
                category_column = self._auto_detect_category_column(data)
                if category_column:
                    logger.info(f"🏷️ Auto-detected category column: {category_column}")
                else:
                    return {
                        'success': False,
                        'error': "Could not find a category column in the data"
                    }

            missing = [c for c in [time_column, value_column, category_column] if c not in data.columns]
            if missing:
                return {
                    'success': False,
                    'error': f"Required columns missing: {', '.join(missing)}"
                }

            df = data.copy()
            # Prepare columns
            df[time_column] = self._prepare_datetime_column(df[time_column])
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            df = df.dropna(subset=[time_column, value_column, category_column])

            if df.empty:
                return {
                    'success': False,
                    'error': 'No valid data after cleaning.'
                }

            # Sort for stability
            df = df.sort_values([category_column, time_column]).reset_index(drop=True)

            # Containers
            anomalies_by_cat = {}
            narratives_by_cat = {}
            methods_by_cat = {}
            stats_by_cat = {}

            # Build combined plot
            fig = go.Figure()

            # Plot one line per category (all points)
            categories = list(df[category_column].dropna().unique())

            total_points = 0
            total_anomalies = 0

            for cat in categories:
                sub = df[df[category_column] == cat].copy()
                # Skip categories with insufficient data
                if len(sub) < 3:
                    methods_by_cat[cat] = 'insufficient_data'
                    stats_by_cat[cat] = {
                        'total_points': len(sub),
                        'anomalies_found': 0,
                        'anomaly_rate': 0
                    }
                    # Still plot the line
                    fig.add_trace(go.Scatter(
                        x=sub[time_column], y=sub[value_column],
                        mode='lines+markers', name=str(cat),
                        line=dict(width=2), marker=dict(size=6)
                    ))
                    continue

                total_points += len(sub)

                # Log data for this category for debugging
                logger.info(f"📊 Category '{cat}': {len(sub)} points")
                mean_val = sub[value_column].mean()
                std_val = sub[value_column].std()
                logger.info(f"   Values: mean={mean_val:.2f}, std={std_val:.2f}")
                logger.info(f"   Min={sub[value_column].min():.2f}, Max={sub[value_column].max():.2f}")

                # Choose detection method based on length
                if len(sub) < 5:
                    is_anomaly, method = self._detect_zscore(sub[value_column])
                else:
                    is_anomaly, method = self._detect_isolation_forest(sub[value_column])
                methods_by_cat[cat] = method

                anomalies = sub[is_anomaly].copy()
                normal = sub[~is_anomaly].copy()

                # Calculate accuracy metrics
                if not anomalies.empty and not normal.empty:
                    # Calculate deviation scores for anomalies
                    anomaly_deviations = []
                    for _, anom in anomalies.iterrows():
                        z_score = abs((anom[value_column] - mean_val) / std_val) if std_val > 0 else 0
                        pct_dev = abs((anom[value_column] - mean_val) / mean_val * 100) if mean_val != 0 else 0
                        anomaly_deviations.append((z_score, pct_dev))
                    
                    avg_z_score = np.mean([d[0] for d in anomaly_deviations])
                    avg_pct_dev = np.mean([d[1] for d in anomaly_deviations])
                    
                    # Calculate separation quality (how far anomalies are from normal data)
                    normal_max_dev = max(abs((normal[value_column] - mean_val) / std_val)) if std_val > 0 else 0
                    separation_score = avg_z_score - normal_max_dev if len(normal) > 0 else avg_z_score
                    
                    logger.info(f"   Detected {len(anomalies)} anomalies using {method}")
                    logger.info(f"   Accuracy Metrics:")
                    logger.info(f"   Average Z-score: {avg_z_score:.2f} (>3.0 = strong anomaly)")
                    logger.info(f"   Average deviation: {avg_pct_dev:.1f}% from mean")
                    logger.info(f"   Separation quality: {separation_score:.2f} (higher = clearer anomalies)")
                    logger.info(f"   Anomaly rate: {len(anomalies)/len(sub)*100:.1f}% of data")
                    
                    for idx, (_, anom) in enumerate(anomalies.iterrows()):
                        z, pct = anomaly_deviations[idx]
                        logger.info(f"   - {anom[time_column].strftime('%Y-%m')}: {anom[value_column]:.2f} (z={z:.2f}, {pct:.1f}% dev)")
                elif not anomalies.empty:
                    logger.info(f"   Detected {len(anomalies)} anomalies using {method} (insufficient normal data for metrics)")
                    for _, anom in anomalies.iterrows():
                        logger.info(f"   - {anom[time_column].strftime('%Y-%m')}: {anom[value_column]:.2f}")
                else:
                    logger.info(f"   No anomalies detected using {method}")

                # Plot line for category
                fig.add_trace(go.Scatter(
                    x=sub[time_column], y=sub[value_column],
                    mode='lines+markers', name=str(cat),
                    line=dict(width=2), marker=dict(size=6)
                ))

                # Compute narratives if any anomalies
                if not anomalies.empty and not normal.empty and normal[value_column].mean() != 0:
                    baseline = normal[value_column].mean()
                    anomalies['deviation'] = ((anomalies[value_column] - baseline) / baseline * 100)
                    narratives = self._generate_narratives(anomalies, time_column)
                    total_anomalies += len(anomalies)
                else:
                    narratives = []
                    anomalies['deviation'] = np.nan

                # Append anomalies as red markers for this category
                if not anomalies.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=anomalies[time_column], y=anomalies[value_column],
                            mode='markers', name=f"Anomalies - {cat}",
                            marker=dict(color='red', size=9, symbol='circle', line=dict(color='darkred', width=2)),
                            hovertemplate=(
                                f"Category: {cat}<br>" +
                                "%{x|%b %Y}<br>" +
                                f"{value_column.replace('_',' ').title()}: " + "%{y:.2f}<extra></extra>"
                            ),
                            showlegend=False
                        )
                    )

                anomalies_by_cat[cat] = anomalies.to_dict('records') if not anomalies.empty else []
                narratives_by_cat[cat] = narratives
                stats_by_cat[cat] = {
                    'total_points': len(sub),
                    'anomalies_found': len(anomalies),
                    'anomaly_rate': (len(anomalies) / len(sub)) if len(sub) else 0
                }

            # Layout for grouped figure
            fig.update_layout(
                title=dict(
                    text=f"Monthly {value_column.replace('_',' ').title()} by Category with Anomalies",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis_title='Month',
                yaxis_title=value_column.replace('_',' ').title(),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='left', x=0),
                margin=dict(t=90, b=80, l=60, r=20),
                template='plotly_white',
                hovermode='x unified',
                height=600
            )
            fig.update_xaxes(tickangle=45, tickformat='%m-%Y', dtick='M2', automargin=True, tickfont=dict(size=11))

            overall_stats = {
                'total_points': total_points,
                'anomalies_found': total_anomalies,
                'anomaly_rate': (total_anomalies / total_points) if total_points else 0
            }

            return {
                'success': True,
                'methods': methods_by_cat,
                'anomalies_by_category': anomalies_by_cat,
                'narratives_by_category': narratives_by_cat,
                'plot': fig,
                'statistics': {
                    'overall': overall_stats,
                    'per_category': stats_by_cat
                }
            }

        except Exception as e:
            logger.error(f"❌ Category anomaly detection error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_datetime_column(self, column):
        """Prepare datetime column by handling timezones and converting to pandas datetime
        
       DateTime column to prepare
            
        Returns: Cleaned datetime column without timezone
        """
        first_value = column.iloc[0] if len(column) > 0 else None
        
        # Handle datetime objects stored in object dtype (from SQL queries)
        if column.dtype == 'object' and isinstance(first_value, datetime):
            # Remove timezone from each datetime object before conversion
            column = column.apply(
                lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) and x.tzinfo is not None else x
            )
            return pd.to_datetime(column, errors='coerce')
        
        # Handle pandas datetime with timezone
        if hasattr(column, 'dt') and hasattr(column.dt, 'tz') and column.dt.tz is not None:
            return column.dt.tz_localize(None)
        
        # Convert to datetime if not already
        return pd.to_datetime(column, errors='coerce')
    
    def _detect_zscore(self, values, threshold=3.0):
        """Detect anomalies using Z-score method
 
                Default 3.0 - 3 standard deviations from mean
            
        """
        z_scores = np.abs(stats.zscore(values))
        is_anomaly = z_scores > threshold
        logger.info(f"   Z-score threshold: {threshold}, max z-score: {z_scores.max():.2f}")
        return is_anomaly, "z-score"
    
    def _detect_isolation_forest(self, values):
        """Detect anomalies using IsolationForest
        """
        contamination_rate = min(self.contamination, 0.5)
        model = IsolationForest(
            contamination=contamination_rate,
            random_state=42
        )
        predictions = model.fit_predict(values.values.reshape(-1, 1))
        is_anomaly = predictions == -1
        logger.info(f"   IsolationForest contamination: {contamination_rate}, anomalies: {is_anomaly.sum()}/{len(is_anomaly)}")
        return is_anomaly, "isolation_forest"
    
    def _generate_narratives(self, anomalies, time_column):
        """Generate human-readable narratives for detected anomalies"""

        narratives = []
        for _, row in anomalies.iterrows():
            month_year = row[time_column].strftime('%B %Y')
            dev = row['deviation']
            direction = "above" if dev > 0 else "below"
            narratives.append(f"{month_year} sales were {abs(dev):.1f}% {direction} usual")
        return narratives
    
    def _generate_plot(self, data, anomalies, time_column, value_column):
        """Generate interactive Plotly chart with anomalies highlighted

        """
        fig = go.Figure()
        
        # Add normal data line
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data[value_column],
            mode='lines+markers',
            name='Normal data',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue')
        ))
        
        # Add anomaly markers
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies[time_column],
                y=anomalies[value_column],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='circle',
                    line=dict(color='darkred', width=2)
                )
            ))
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text=f'Monthly {value_column.replace("_", " ").title()} with Anomalies',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Month',
            yaxis_title=value_column.replace("_", " ").title(),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.05,
                xanchor='left',
                x=0
            ),
            margin=dict(t=90, b=80, l=60, r=20),
            template='plotly_white',
            hovermode='x unified',
            height=400
        )
        
        # Configure x-axis
        fig.update_xaxes(
            tickangle=45,
            tickformat='%m-%Y',
            dtick='M2',
            automargin=True,
            tickfont=dict(size=11)
        )
        
        return fig
    
    def summarize_anomalies(self, result):
        """Generate a human-readable summary of anomaly detection results"""
        if not result.get('success'):
            return f"Analysis failed: {result.get('error', 'Unknown error')}"
        
        if 'message' in result:
            return result['message']
        # Handle grouped/category results
        if 'anomalies_by_category' in result or 'narratives_by_category' in result:
            stats = result.get('statistics', {})
            overall = stats.get('overall', {})
            per_cat = stats.get('per_category', {})
            methods = result.get('methods', {})

            lines = []
            if overall:
                lines.append(
                    f"Found {overall.get('anomalies_found', 0)} anomalies "
                    f"({overall.get('anomaly_rate', 0)*100:.1f}% of points) across categories"
                )
            # Per-category breakdown
            for cat, cat_stats in per_cat.items():
                method = methods.get(cat, 'n/a')
                lines.append(
                    f"- {cat}: {cat_stats.get('anomalies_found', 0)} anomalies "
                    f"out of {cat_stats.get('total_points', 0)} points using {method}"
                )
                narratives = result.get('narratives_by_category', {}).get(cat, [])
                for n in narratives:
                    lines.append(f"  • {n}")
            return "\n".join(lines)

        # Default single-series summary
        summary = [
            f"Analysis Method: {result['method']}",
            f"Found {result['statistics']['anomalies_found']} anomalies "
            f"({result['statistics']['anomaly_rate']*100:.1f}% of data points)"
        ]
        
        if result.get('narratives'):
            summary.append("\nAnomalies detected:")
            summary.extend([f"- {narrative}" for narrative in result['narratives']])
        
        return "\n".join(summary)
