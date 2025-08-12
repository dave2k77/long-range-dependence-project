"""
Data Quality Assessment Module for Long-Range Dependence Analysis

This module provides comprehensive data quality assessment for time series data:
- Missing value analysis
- Outlier detection
- Data distribution analysis
- Time series specific checks
- Quality scoring and reporting
- Data validation for analysis requirements
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List, Tuple, Any
import warnings
from scipy import stats
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class DataQualityChecker:
    """
    A comprehensive data quality checker for time series data.
    
    Performs various quality assessments and provides detailed reports
    for long-range dependence analysis.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the quality checker.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print informative messages during quality checks
        """
        self.verbose = verbose
        self.quality_scores = {}
        self.issues = []
    
    def assess_quality(self, df: pd.DataFrame, 
                      target_column: Optional[str] = None,
                      min_length: int = 100,
                      max_missing_pct: float = 0.1,
                      outlier_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of time series data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Time series data to assess
        target_column : Optional[str]
            Specific column to analyze (if None, analyzes all numeric columns)
        min_length : int
            Minimum required length for analysis
        max_missing_pct : float
            Maximum acceptable percentage of missing values
        outlier_threshold : float
            Threshold for outlier detection (in standard deviations)
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive quality assessment report
        """
        if self.verbose:
            print("Performing comprehensive data quality assessment...")
        
        # Reset previous results
        self.quality_scores = {}
        self.issues = []
        
        # Select columns to analyze
        if target_column is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in data")
            columns_to_analyze = numeric_columns
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            columns_to_analyze = [target_column]
        
        # Perform quality checks
        results = {
            'overall_score': 0.0,
            'columns': {},
            'summary': {},
            'issues': [],
            'recommendations': []
        }
        
        total_score = 0.0
        num_columns = len(columns_to_analyze)
        
        for column in columns_to_analyze:
            column_results = self._assess_column_quality(
                df[column], column, min_length, max_missing_pct, outlier_threshold
            )
            results['columns'][column] = column_results
            total_score += column_results['quality_score']
        
        # Calculate overall score
        results['overall_score'] = total_score / num_columns
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(df, columns_to_analyze)
        
        # Collect all issues
        for column_result in results['columns'].values():
            results['issues'].extend(column_result['issues'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Store results
        self.quality_scores = results
        
        if self.verbose:
            print(f"Quality assessment complete. Overall score: {results['overall_score']:.2f}/10")
        
        return results
    
    def _assess_column_quality(self, series: pd.Series, column_name: str,
                              min_length: int, max_missing_pct: float,
                              outlier_threshold: float) -> Dict[str, Any]:
        """Assess quality of a single column."""
        results = {
            'column_name': column_name,
            'quality_score': 0.0,
            'issues': [],
            'statistics': {},
            'checks': {}
        }
        
        # Basic statistics
        results['statistics'] = self._calculate_statistics(series)
        
        # Perform quality checks
        checks = {}
        
        # Length check
        length_check = self._check_length(series, min_length)
        checks['length'] = length_check
        
        # Missing values check
        missing_check = self._check_missing_values(series, max_missing_pct)
        checks['missing_values'] = missing_check
        
        # Outlier check
        outlier_check = self._check_outliers(series, outlier_threshold)
        checks['outliers'] = outlier_check
        
        # Distribution check
        distribution_check = self._check_distribution(series)
        checks['distribution'] = distribution_check
        
        # Time series specific checks
        if series.index.dtype == 'datetime64[ns]':
            time_check = self._check_time_series_properties(series)
            checks['time_series'] = time_check
        
        # Stationarity check
        stationarity_check = self._check_stationarity(series)
        checks['stationarity'] = stationarity_check
        
        results['checks'] = checks
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(checks)
        results['quality_score'] = quality_score
        
        # Collect issues
        for check_name, check_result in checks.items():
            if not check_result['passed']:
                results['issues'].append({
                    'check': check_name,
                    'severity': check_result['severity'],
                    'message': check_result['message']
                })
        
        return results
    
    def _calculate_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistics for the series."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan
            }
        
        return {
            'count': len(clean_series),
            'mean': clean_series.mean(),
            'std': clean_series.std(),
            'min': clean_series.min(),
            'max': clean_series.max(),
            'skewness': skew(clean_series),
            'kurtosis': kurtosis(clean_series),
            'q25': clean_series.quantile(0.25),
            'q50': clean_series.quantile(0.50),
            'q75': clean_series.quantile(0.75)
        }
    
    def _check_length(self, series: pd.Series, min_length: int) -> Dict[str, Any]:
        """Check if series meets minimum length requirement."""
        actual_length = len(series.dropna())
        
        passed = actual_length >= min_length
        severity = 'high' if not passed else 'none'
        
        return {
            'passed': passed,
            'severity': severity,
            'actual_length': actual_length,
            'required_length': min_length,
            'message': f"Length: {actual_length}/{min_length} required"
        }
    
    def _check_missing_values(self, series: pd.Series, max_missing_pct: float) -> Dict[str, Any]:
        """Check for missing values."""
        missing_count = series.isnull().sum()
        missing_pct = missing_count / len(series)
        
        passed = missing_pct <= max_missing_pct
        severity = 'high' if missing_pct > 0.2 else 'medium' if missing_pct > 0.05 else 'low'
        
        return {
            'passed': passed,
            'severity': severity,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'max_allowed_pct': max_missing_pct,
            'message': f"Missing: {missing_count} ({missing_pct:.1%})"
        }
    
    def _check_outliers(self, series: pd.Series, threshold: float) -> Dict[str, Any]:
        """Check for outliers using z-score method."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'passed': False,
                'severity': 'high',
                'outlier_count': 0,
                'outlier_pct': 0.0,
                'message': "No valid data to check for outliers"
            }
        
        z_scores = np.abs(stats.zscore(clean_series))
        outliers = z_scores > threshold
        outlier_count = outliers.sum()
        outlier_pct = outlier_count / len(clean_series)
        
        # Determine severity based on outlier percentage
        if outlier_pct > 0.1:
            severity = 'high'
        elif outlier_pct > 0.05:
            severity = 'medium'
        else:
            severity = 'low'
        
        passed = outlier_pct <= 0.1  # Consider acceptable if <= 10%
        
        return {
            'passed': passed,
            'severity': severity,
            'outlier_count': outlier_count,
            'outlier_pct': outlier_pct,
            'threshold': threshold,
            'message': f"Outliers: {outlier_count} ({outlier_pct:.1%})"
        }
    
    def _check_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Check distribution properties."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'passed': False,
                'severity': 'high',
                'message': "No valid data to check distribution"
            }
        
        # Check for normality using Shapiro-Wilk test
        try:
            shapiro_stat, shapiro_p = stats.shapiro(clean_series)
            is_normal = shapiro_p > 0.05
        except:
            is_normal = False
            shapiro_p = 0.0
        
        # Check for extreme skewness or kurtosis
        skewness = skew(clean_series)
        kurt = kurtosis(clean_series)
        
        extreme_skew = abs(skewness) > 2
        extreme_kurt = abs(kurt) > 7
        
        issues = []
        if extreme_skew:
            issues.append(f"Extreme skewness: {skewness:.2f}")
        if extreme_kurt:
            issues.append(f"Extreme kurtosis: {kurt:.2f}")
        
        passed = not extreme_skew and not extreme_kurt
        severity = 'medium' if len(issues) > 0 else 'none'
        
        return {
            'passed': passed,
            'severity': severity,
            'is_normal': is_normal,
            'shapiro_p': shapiro_p,
            'skewness': skewness,
            'kurtosis': kurt,
            'issues': issues,
            'message': f"Distribution: skew={skewness:.2f}, kurt={kurt:.2f}"
        }
    
    def _check_time_series_properties(self, series: pd.Series) -> Dict[str, Any]:
        """Check time series specific properties."""
        clean_series = series.dropna()
        
        if len(clean_series) < 2:
            return {
                'passed': False,
                'severity': 'high',
                'message': "Insufficient data for time series analysis"
            }
        
        # Check for regular time intervals
        time_diffs = clean_series.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            time_diff_std = time_diffs.std()
            time_diff_mean = time_diffs.mean()
            irregular_intervals = time_diff_std / time_diff_mean > 0.1
        else:
            irregular_intervals = False
        
        # Check for gaps in time series
        expected_length = (clean_series.index[-1] - clean_series.index[0]) / time_diffs.mean() + 1
        actual_length = len(clean_series)
        gap_ratio = actual_length / expected_length if expected_length > 0 else 1.0
        
        issues = []
        if irregular_intervals:
            issues.append("Irregular time intervals")
        if gap_ratio < 0.9:
            issues.append(f"Large gaps in time series (ratio: {gap_ratio:.2f})")
        
        passed = len(issues) == 0
        severity = 'medium' if len(issues) > 0 else 'none'
        
        return {
            'passed': passed,
            'severity': severity,
            'irregular_intervals': irregular_intervals,
            'gap_ratio': gap_ratio,
            'issues': issues,
            'message': f"Time series: {len(issues)} issues found"
        }
    
    def _check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Check stationarity using ADF test."""
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {
                'passed': False,
                'severity': 'high',
                'message': "Insufficient data for stationarity test"
            }
        
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_p, _, _, _, _ = adfuller(clean_series)
            is_stationary = adf_p < 0.05
        except:
            adf_p = 1.0
            is_stationary = False
        
        severity = 'medium' if not is_stationary else 'none'
        
        return {
            'passed': is_stationary,
            'severity': severity,
            'adf_statistic': adf_stat if 'adf_stat' in locals() else np.nan,
            'adf_pvalue': adf_p,
            'is_stationary': is_stationary,
            'message': f"Stationarity: {'stationary' if is_stationary else 'non-stationary'} (p={adf_p:.4f})"
        }
    
    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """Calculate overall quality score based on checks."""
        score = 10.0  # Start with perfect score
        
        # Deduct points for failed checks
        for check_name, check_result in checks.items():
            if not check_result['passed']:
                if check_result['severity'] == 'high':
                    score -= 3.0
                elif check_result['severity'] == 'medium':
                    score -= 1.5
                elif check_result['severity'] == 'low':
                    score -= 0.5
        
        return max(0.0, score)
    
    def _generate_summary(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'analyzed_columns': len(columns),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'date_range': {
                'start': df.index.min() if df.index.dtype == 'datetime64[ns]' else None,
                'end': df.index.max() if df.index.dtype == 'datetime64[ns]' else None
            }
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        overall_score = results['overall_score']
        
        if overall_score < 5.0:
            recommendations.append("Data quality is poor. Consider data cleaning or finding alternative data sources.")
        elif overall_score < 7.0:
            recommendations.append("Data quality is moderate. Apply preprocessing before analysis.")
        else:
            recommendations.append("Data quality is good. Proceed with analysis.")
        
        # Check for specific issues
        for issue in results['issues']:
            if issue['check'] == 'missing_values' and issue['severity'] == 'high':
                recommendations.append("High percentage of missing values. Consider imputation or data collection.")
            
            elif issue['check'] == 'outliers' and issue['severity'] == 'high':
                recommendations.append("High number of outliers detected. Consider outlier treatment methods.")
            
            elif issue['check'] == 'stationarity' and issue['severity'] == 'medium':
                recommendations.append("Series is non-stationary. Consider differencing or detrending.")
            
            elif issue['check'] == 'length' and issue['severity'] == 'high':
                recommendations.append("Insufficient data length. Consider collecting more data or using different methods.")
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a detailed quality report."""
        if not self.quality_scores:
            raise ValueError("No quality assessment results available. Run assess_quality() first.")
        
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        results = self.quality_scores
        report.append("OVERALL SUMMARY")
        report.append("-" * 20)
        report.append(f"Overall Quality Score: {results['overall_score']:.2f}/10")
        report.append(f"Total Issues Found: {len(results['issues'])}")
        report.append(f"Columns Analyzed: {len(results['columns'])}")
        report.append("")
        
        # Dataset summary
        summary = results['summary']
        report.append("DATASET SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Rows: {summary['total_rows']:,}")
        report.append(f"Total Columns: {summary['total_columns']}")
        report.append(f"Memory Usage: {summary['memory_usage'] / 1024:.1f} KB")
        if summary['date_range']['start']:
            report.append(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        report.append("")
        
        # Column details
        report.append("COLUMN DETAILS")
        report.append("-" * 20)
        for column_name, column_result in results['columns'].items():
            report.append(f"\n{column_name}:")
            report.append(f"  Quality Score: {column_result['quality_score']:.2f}/10")
            report.append(f"  Issues: {len(column_result['issues'])}")
            
            # Statistics
            stats = column_result['statistics']
            report.append(f"  Statistics:")
            report.append(f"    Count: {stats['count']:,}")
            report.append(f"    Mean: {stats['mean']:.4f}")
            report.append(f"    Std: {stats['std']:.4f}")
            report.append(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Issues summary
        if results['issues']:
            report.append("\nDETAILED ISSUES")
            report.append("-" * 20)
            for i, issue in enumerate(results['issues'], 1):
                report.append(f"{i}. {issue['check'].upper()}: {issue['message']} (Severity: {issue['severity']})")
        
        # Recommendations
        if results['recommendations']:
            report.append("\nRECOMMENDATIONS")
            report.append("-" * 20)
            for i, rec in enumerate(results['recommendations'], 1):
                report.append(f"{i}. {rec}")
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            if self.verbose:
                print(f"Quality report saved to: {output_file}")
        
        return report_text
    
    def plot_quality_summary(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Create visual summary of quality assessment."""
        if not self.quality_scores:
            raise ValueError("No quality assessment results available. Run assess_quality() first.")
        
        results = self.quality_scores
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Data Quality Assessment Summary', fontsize=16)
        
        # Overall quality score
        ax1 = axes[0, 0]
        score = results['overall_score']
        ax1.bar(['Overall Score'], [score], color='skyblue' if score >= 7 else 'orange' if score >= 5 else 'red')
        ax1.set_ylim(0, 10)
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Overall Quality Score')
        
        # Issues by severity
        ax2 = axes[0, 1]
        severity_counts = {}
        for issue in results['issues']:
            severity = issue['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = ['red' if s == 'high' else 'orange' if s == 'medium' else 'yellow' for s in severities]
            ax2.bar(severities, counts, color=colors)
            ax2.set_ylabel('Number of Issues')
            ax2.set_title('Issues by Severity')
        
        # Column quality scores
        ax3 = axes[1, 0]
        columns = list(results['columns'].keys())
        scores = [results['columns'][col]['quality_score'] for col in columns]
        colors = ['green' if s >= 7 else 'orange' if s >= 5 else 'red' for s in scores]
        ax3.bar(columns, scores, color=colors)
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Quality Scores by Column')
        ax3.tick_params(axis='x', rotation=45)
        
        # Check results
        ax4 = axes[1, 1]
        check_results = {}
        for col_result in results['columns'].values():
            for check_name, check_result in col_result['checks'].items():
                if check_name not in check_results:
                    check_results[check_name] = {'passed': 0, 'failed': 0}
                if check_result['passed']:
                    check_results[check_name]['passed'] += 1
                else:
                    check_results[check_name]['failed'] += 1
        
        if check_results:
            checks = list(check_results.keys())
            passed = [check_results[c]['passed'] for c in checks]
            failed = [check_results[c]['failed'] for c in checks]
            
            x = np.arange(len(checks))
            ax4.bar(x - 0.2, passed, 0.4, label='Passed', color='green')
            ax4.bar(x + 0.2, failed, 0.4, label='Failed', color='red')
            ax4.set_xlabel('Quality Checks')
            ax4.set_ylabel('Number of Columns')
            ax4.set_title('Quality Check Results')
            ax4.set_xticks(x)
            ax4.set_xticklabels(checks, rotation=45)
            ax4.legend()
        
        plt.tight_layout()
        plt.show()


# Convenience functions
def assess_data_quality(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Assess data quality with default settings."""
    checker = DataQualityChecker()
    return checker.assess_quality(df, **kwargs)


def generate_quality_report(df: pd.DataFrame, output_file: Optional[str] = None, **kwargs) -> str:
    """Generate a comprehensive quality report."""
    checker = DataQualityChecker()
    checker.assess_quality(df, **kwargs)
    return checker.generate_report(output_file)


def plot_quality_summary(df: pd.DataFrame, **kwargs) -> None:
    """Create visual quality summary."""
    checker = DataQualityChecker()
    checker.assess_quality(df, **kwargs)
    checker.plot_quality_summary()


def validate_for_analysis(df: pd.DataFrame, min_length: int = 100, 
                         max_missing_pct: float = 0.1) -> bool:
    """Quick validation for analysis suitability."""
    checker = DataQualityChecker(verbose=False)
    results = checker.assess_quality(df, min_length=min_length, 
                                   max_missing_pct=max_missing_pct)
    return results['overall_score'] >= 6.0
