"""
Dataset Submission Module

This module handles the submission, validation, and testing of new datasets
for long-range dependence analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import shutil
from datetime import datetime

from .standards import DatasetStandards, ValidationResult, ComplianceChecker, SubmissionStatus


@dataclass
class DatasetMetadata:
    """Metadata for a submitted dataset"""
    name: str
    version: str
    author: str
    description: str
    source: str
    format: str  # e.g., "csv", "json", "parquet"
    size: int  # number of rows
    columns: List[str]
    file_path: str
    sampling_frequency: Optional[str] = None
    units: Optional[str] = None
    collection_date: Optional[str] = None
    metadata_file: Optional[str] = None
    quality_report: Optional[str] = None
    submission_date: Optional[str] = None
    status: str = "pending"


class DatasetValidator:
    """Validator for submitted datasets"""
    
    def __init__(self):
        self.compliance_checker = ComplianceChecker()
        self.standards = DatasetStandards()
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate a dataset"""
        return self.compliance_checker.check_dataset_compliance(data)
    
    def validate_dataset_file(self, file_path: str) -> ValidationResult:
        """Validate dataset file format and accessibility"""
        return self.compliance_checker.validate_dataset_format(file_path)
    
    def validate_dataset_content(self, data: pd.DataFrame) -> ValidationResult:
        """Validate dataset content against quality requirements"""
        return self.compliance_checker.validate_dataset_content(data)
    
    def validate_metadata(self, metadata: DatasetMetadata) -> ValidationResult:
        """Validate dataset metadata completeness"""
        requirements = self.standards.get_quality_requirements()
        missing_fields = []
        
        # Check required metadata fields
        for field in requirements["required_metadata"]:
            if not hasattr(metadata, field) or getattr(metadata, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=[f"Missing required metadata fields: {missing_fields}"],
                warnings=[],
                details={"missing_fields": missing_fields}
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=[],
            warnings=[],
            details={"checked_fields": requirements["required_metadata"]}
        )
    
    def perform_quality_checks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive quality checks on the dataset"""
        quality_report = {
            "basic_stats": {},
            "missing_values": {},
            "outliers": {},
            "stationarity": {},
            "distribution": {},
            "quality_score": 0.0
        }
        
        # Basic statistics
        if "value" in data.columns:
            values = data["value"].dropna()
            quality_report["basic_stats"] = {
                "length": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "skewness": float(values.skew()),
                "kurtosis": float(values.kurtosis())
            }
        
        # Missing values analysis
        quality_report["missing_values"] = {
            "total_missing": int(data.isnull().sum().sum()),
            "missing_ratio": float(data.isnull().sum().sum() / (len(data) * len(data.columns))),
            "missing_by_column": data.isnull().sum().to_dict()
        }
        
        # Outlier detection
        if "value" in data.columns:
            values = data["value"].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
            
            quality_report["outliers"] = {
                "count": int(len(outliers)),
                "ratio": float(len(outliers) / len(values)),
                "outlier_values": outliers.tolist()[:10]  # First 10 outliers
            }
        
        # Stationarity test (simplified)
        if "value" in data.columns and len(data) > 50:
            values = data["value"].dropna()
            # Simple stationarity check: variance of first and second half
            mid = len(values) // 2
            var1 = values[:mid].var()
            var2 = values[mid:].var()
            var_ratio = max(var1, var2) / min(var1, var2) if min(var1, var2) > 0 else float('inf')
            
            quality_report["stationarity"] = {
                "variance_ratio": float(var_ratio),
                "is_stationary": var_ratio < 2.0  # Simple threshold
            }
        
        # Distribution analysis
        if "value" in data.columns:
            values = data["value"].dropna()
            quality_report["distribution"] = {
                "normality_test": self._test_normality(values),
                "has_negative_values": bool((values < 0).any()),
                "has_zero_values": bool((values == 0).any())
            }
        
        # Calculate overall quality score
        quality_report["quality_score"] = self._calculate_quality_score(quality_report)
        
        return quality_report
    
    def _test_normality(self, values: pd.Series) -> Dict[str, Any]:
        """Perform normality test on the data"""
        try:
            from scipy import stats
            statistic, p_value = stats.shapiro(values)
            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05
            }
        except ImportError:
            return {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "error": "scipy not available"
            }
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        score = 1.0
        
        # Penalize missing values
        missing_ratio = quality_report["missing_values"]["missing_ratio"]
        score -= missing_ratio * 0.5
        
        # Penalize outliers
        outlier_ratio = quality_report["outliers"]["ratio"]
        score -= outlier_ratio * 0.3
        
        # Penalize non-stationarity
        if "stationarity" in quality_report and "is_stationary" in quality_report["stationarity"]:
            if not quality_report["stationarity"]["is_stationary"]:
                score -= 0.1
        
        # Penalize non-normality
        if "distribution" in quality_report and "normality_test" in quality_report["distribution"]:
            normality = quality_report["distribution"]["normality_test"]
            if normality.get("is_normal") is False:
                score -= 0.05
        
        return max(0.0, min(1.0, score))


class DatasetTester:
    """Tester for submitted datasets"""
    
    def __init__(self):
        self.test_models = self._get_test_models()
    
    def _get_test_models(self) -> Dict[str, Any]:
        """Get test models for dataset validation"""
        # This would typically import actual analysis models
        # For now, we'll create simple test functions
        return {
            "basic_stats": self._test_basic_stats,
            "lrd_analysis": self._test_lrd_analysis,
            "robustness": self._test_robustness
        }
    
    def _test_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test basic statistical properties"""
        if "value" not in data.columns:
            return {"error": "No 'value' column found"}
        
        values = data["value"].dropna()
        
        return {
            "length": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "autocorrelation_lag1": float(values.autocorr(lag=1)) if len(values) > 1 else None,
            "success": True
        }
    
    def _test_lrd_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test long-range dependence analysis"""
        if "value" not in data.columns:
            return {"error": "No 'value' column found"}
        
        values = data["value"].dropna()
        
        if len(values) < 100:
            return {"error": "Insufficient data for LRD analysis"}
        
        try:
            # Simple variance-time plot analysis
            n = len(values)
            scales = [2**i for i in range(2, int(np.log2(n/4)))]
            variances = []
            
            for scale in scales:
                if scale < n:
                    # Calculate variance at this scale
                    chunks = [values[i:i+scale] for i in range(0, n-scale+1, scale)]
                    chunk_vars = [chunk.var() for chunk in chunks if len(chunk) == scale]
                    if chunk_vars:
                        variances.append(np.mean(chunk_vars))
                    else:
                        variances.append(np.nan)
                else:
                    variances.append(np.nan)
            
            # Remove NaN values
            valid_scales = [s for s, v in zip(scales, variances) if not np.isnan(v)]
            valid_variances = [v for v in variances if not np.isnan(v)]
            
            if len(valid_scales) < 3:
                return {"error": "Insufficient valid scales for analysis"}
            
            # Fit power law: variance ~ scale^beta
            log_scales = np.log(valid_scales)
            log_variances = np.log(valid_variances)
            
            # Simple linear regression
            coeffs = np.polyfit(log_scales, log_variances, 1)
            beta = coeffs[0]
            hurst_estimate = 1 - beta / 2
            
            return {
                "hurst_estimate": float(hurst_estimate),
                "beta_estimate": float(beta),
                "r_squared": float(self._calculate_r_squared(log_scales, log_variances, coeffs)),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"LRD analysis failed: {str(e)}"}
    
    def _test_robustness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test robustness of the dataset"""
        if "value" not in data.columns:
            return {"error": "No 'value' column found"}
        
        values = data["value"].dropna()
        
        # Test with different subsets
        results = {}
        
        # Test with 50% of data
        subset_50 = values.sample(frac=0.5, random_state=42)
        results["subset_50"] = {
            "length": len(subset_50),
            "mean": float(subset_50.mean()),
            "std": float(subset_50.std())
        }
        
        # Test with 25% of data
        subset_25 = values.sample(frac=0.25, random_state=42)
        results["subset_25"] = {
            "length": len(subset_25),
            "mean": float(subset_25.mean()),
            "std": float(subset_25.std())
        }
        
        # Calculate stability metrics
        original_mean = values.mean()
        original_std = values.std()
        
        stability_metrics = {
            "mean_stability_50": abs(results["subset_50"]["mean"] - original_mean) / abs(original_mean) if original_mean != 0 else float('inf'),
            "std_stability_50": abs(results["subset_50"]["std"] - original_std) / original_std if original_std != 0 else float('inf'),
            "mean_stability_25": abs(results["subset_25"]["mean"] - original_mean) / abs(original_mean) if original_mean != 0 else float('inf'),
            "std_stability_25": abs(results["subset_25"]["std"] - original_std) / original_std if original_std != 0 else float('inf')
        }
        
        return {
            "subset_results": results,
            "stability_metrics": stability_metrics,
            "success": True
        }
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Calculate R-squared for linear fit"""
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test a dataset"""
        results = {}
        
        for test_name, test_func in self.test_models.items():
            try:
                results[test_name] = test_func(data)
            except Exception as e:
                results[test_name] = {"error": str(e), "success": False}
        
        return results
    
    def evaluate_dataset_quality(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall dataset quality based on test results"""
        quality_metrics = {
            "overall_score": 0.0,
            "test_success_rate": 0.0,
            "lrd_detected": False,
            "stability_score": 0.0,
            "recommendations": []
        }
        
        successful_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            if result.get("success", False):
                successful_tests += 1
                
                if test_name == "lrd_analysis":
                    hurst_est = result.get("hurst_estimate", 0.5)
                    if 0.5 < hurst_est < 1.0:
                        quality_metrics["lrd_detected"] = True
                        quality_metrics["overall_score"] += 0.3
                
                elif test_name == "robustness":
                    stability = result.get("stability_metrics", {})
                    if stability:
                        # Calculate average stability
                        stability_values = [v for v in stability.values() if v != float('inf')]
                        if stability_values:
                            avg_stability = np.mean(stability_values)
                            quality_metrics["stability_score"] = max(0, 1 - avg_stability)
                            quality_metrics["overall_score"] += quality_metrics["stability_score"] * 0.2
        
        quality_metrics["test_success_rate"] = successful_tests / total_tests if total_tests > 0 else 0.0
        quality_metrics["overall_score"] += quality_metrics["test_success_rate"] * 0.5
        
        # Generate recommendations
        if quality_metrics["test_success_rate"] < 0.8:
            quality_metrics["recommendations"].append("Dataset has low test success rate")
        
        if not quality_metrics["lrd_detected"]:
            quality_metrics["recommendations"].append("No clear long-range dependence detected")
        
        if quality_metrics["stability_score"] < 0.7:
            quality_metrics["recommendations"].append("Dataset shows low stability across subsets")
        
        return quality_metrics


class DatasetRegistry:
    """Registry for managing submitted datasets"""
    
    def __init__(self, registry_path: str = "datasets/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the dataset registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"datasets": {}, "metadata": {"version": "1.0"}}
    
    def _save_registry(self):
        """Save the dataset registry"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register(self, metadata: DatasetMetadata) -> bool:
        """Register a new dataset (alias for register_dataset)"""
        return self.register_dataset(metadata)
    
    def register_dataset(self, metadata: DatasetMetadata) -> bool:
        """Register a new dataset"""
        dataset_id = f"{metadata.name}_{metadata.version}"
        
        if dataset_id in self.registry["datasets"]:
            return False  # Dataset already exists
        
        # Convert to dict for JSON serialization
        metadata_dict = {
            "name": metadata.name,
            "version": metadata.version,
            "author": metadata.author,
            "description": metadata.description,
            "source": metadata.source,
            "format": metadata.format,
            "size": metadata.size,
            "columns": metadata.columns,
            "file_path": metadata.file_path,
            "sampling_frequency": metadata.sampling_frequency,
            "units": metadata.units,
            "collection_date": metadata.collection_date,
            "metadata_file": metadata.metadata_file,
            "quality_report": metadata.quality_report,
            "submission_date": metadata.submission_date,
            "status": metadata.status
        }
        
        self.registry["datasets"][dataset_id] = metadata_dict
        self._save_registry()
        return True
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get a dataset by ID"""
        if dataset_id in self.registry["datasets"]:
            return DatasetMetadata(**self.registry["datasets"][dataset_id])
        return None
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets"""
        return list(self.registry["datasets"].keys())
    
    def update_dataset_status(self, dataset_id: str, status: str) -> bool:
        """Update dataset status"""
        if dataset_id in self.registry["datasets"]:
            self.registry["datasets"][dataset_id]["status"] = status
            self._save_registry()
            return True
        return False


class DatasetSubmission:
    """Main class for handling dataset submissions"""
    
    def __init__(self, registry_path: str = "datasets/registry.json"):
        self.validator = DatasetValidator()
        self.tester = DatasetTester()
        self.registry = DatasetRegistry(registry_path)
    
    def submit_dataset(self, 
                      file_path: str,
                      metadata: DatasetMetadata,
                      test_dataset: bool = True) -> Dict[str, Any]:
        """Submit a new dataset for validation and testing"""
        submission_result = {
            "success": False,
            "validation_results": [],
            "quality_report": None,
            "test_results": None,
            "quality_evaluation": None,
            "dataset_id": None,
            "message": ""
        }
        
        try:
            # Step 1: Validate file format
            format_validation = self.validator.validate_dataset_file(file_path)
            submission_result["validation_results"].append(format_validation)
            
            if not format_validation.is_valid:
                submission_result["message"] = "Dataset file validation failed"
                return submission_result
            
            # Step 2: Load and validate data content
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    data = pd.read_json(file_path)
                elif file_path.endswith('.parquet'):
                    data = pd.read_parquet(file_path)
                else:
                    submission_result["message"] = "Unsupported file format"
                    return submission_result
            except Exception as e:
                submission_result["message"] = f"Failed to load dataset: {str(e)}"
                return submission_result
            
            content_validation = self.validator.validate_dataset_content(data)
            submission_result["validation_results"].append(content_validation)
            
            if not content_validation.is_valid:
                submission_result["message"] = "Dataset content validation failed"
                return submission_result
            
            # Step 3: Validate metadata
            metadata_validation = self.validator.validate_metadata(metadata)
            submission_result["validation_results"].append(metadata_validation)
            
            if not metadata_validation.is_valid:
                submission_result["message"] = "Dataset metadata validation failed"
                return submission_result
            
            # Step 4: Generate quality report
            quality_report = self.validator.perform_quality_checks(data)
            submission_result["quality_report"] = quality_report
            
            # Check quality score
            if quality_report["quality_score"] < 0.7:
                submission_result["message"] = "Dataset quality below threshold"
                return submission_result
            
            # Step 5: Test dataset if requested
            if test_dataset:
                test_results = self.tester.test(data)
                quality_evaluation = self.tester.evaluate_dataset_quality(test_results)
                
                submission_result["test_results"] = test_results
                submission_result["quality_evaluation"] = quality_evaluation
                
                # Check if quality meets standards
                if quality_evaluation["overall_score"] < 0.6:
                    submission_result["message"] = "Dataset quality evaluation below threshold"
                    return submission_result
            
            # Step 6: Copy file to datasets directory
            datasets_dir = Path("datasets/submitted")
            datasets_dir.mkdir(parents=True, exist_ok=True)
            
            new_file_path = datasets_dir / f"{metadata.name}_{metadata.version}{Path(file_path).suffix}"
            shutil.copy2(file_path, new_file_path)
            
            # Step 7: Save quality report
            quality_report_path = datasets_dir / f"{metadata.name}_{metadata.version}_quality.json"
            
            # Convert quality report to JSON-serializable format
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (bool, np.bool_)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_quality_report = convert_to_serializable(quality_report)
            
            with open(quality_report_path, 'w') as f:
                json.dump(serializable_quality_report, f, indent=2)
            
            # Step 8: Update metadata
            metadata.file_path = str(new_file_path)
            metadata.quality_report = str(quality_report_path)
            metadata.submission_date = datetime.now().isoformat()
            metadata.size_mb = os.path.getsize(new_file_path) / (1024 * 1024)
            metadata.length = len(data)
            metadata.columns = list(data.columns)
            
            # Step 9: Register dataset
            dataset_id = f"{metadata.name}_{metadata.version}"
            
            if self.registry.register_dataset(metadata):
                submission_result["dataset_id"] = dataset_id
                submission_result["success"] = True
                submission_result["message"] = "Dataset submitted successfully"
            else:
                submission_result["message"] = "Dataset already exists in registry"
            
        except Exception as e:
            submission_result["message"] = f"Submission failed: {str(e)}"
        
        return submission_result
    
    def get_submission_status(self, dataset_id: str) -> Optional[str]:
        """Get the status of a submitted dataset"""
        dataset = self.registry.get_dataset(dataset_id)
        return dataset.status if dataset else None
    
    def update_submission_status(self, dataset_id: str, status: str) -> bool:
        """Update the status of a submitted dataset"""
        return self.registry.update_dataset_status(dataset_id, status)
