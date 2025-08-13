"""
Standards and Validation for Model and Dataset Submissions

This module defines the standards that submitted models and datasets must meet
to be accepted into the benchmark.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path


class SubmissionStatus(Enum):
    """Status of a submission"""
    PENDING = "pending"
    VALIDATING = "validating"
    TESTING = "testing"
    APPROVED = "approved"
    REJECTED = "rejected"
    INTEGRATED = "integrated"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    status: SubmissionStatus
    errors: List[str]
    warnings: List[str]
    details: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class ModelStandards:
    """Standards that submitted models must meet"""
    
    @staticmethod
    def get_required_interface() -> List[str]:
        """Get the required interface for estimator models"""
        return [
            "fit",
            "estimate_hurst",
            "estimate_alpha",
            "get_confidence_intervals",
            "get_quality_metrics"
        ]
    
    @staticmethod
    def get_parameter_constraints() -> Dict[str, Any]:
        """Get parameter constraints for models"""
        return {
            "hurst_range": [0.0, 1.0],
            "alpha_range": [0.0, 2.0],
            "beta_range": [0.0, 2.0],
            "min_data_length": 100,
            "max_data_length": 100000,
            "required_parameters": ["data"],
            "optional_parameters": ["scales", "moments", "wavelet_type"]
        }
    
    @staticmethod
    def get_quality_thresholds() -> Dict[str, float]:
        """Get quality thresholds for model performance"""
        return {
            "min_r_squared": 0.7,
            "max_std_error": 0.2,
            "min_convergence_rate": 0.8,
            "max_computation_time": 300.0,  # seconds
            "min_memory_efficiency": 0.5
        }
    
    @staticmethod
    def get_documentation_requirements() -> List[str]:
        """Get documentation requirements for models"""
        return [
            "README.md with usage examples",
            "API documentation",
            "Parameter descriptions",
            "Performance characteristics",
            "Limitations and assumptions",
            "References to original papers"
        ]


class DatasetStandards:
    """Standards that submitted datasets must meet"""
    
    @staticmethod
    def get_format_requirements() -> Dict[str, Any]:
        """Get format requirements for datasets"""
        return {
            "supported_formats": ["csv", "json", "parquet", "hdf5"],
            "required_columns": ["timestamp", "value"],
            "min_length": 100,
            "max_length": 100000,
            "encoding": "utf-8",
            "delimiter": ",",
            "data_types": {
                "timestamp": ["datetime64[ns]", "int64", "float64"],
                "value": ["float64", "float32"]
            }
        }
    
    @staticmethod
    def get_quality_requirements() -> Dict[str, Any]:
        """Get quality requirements for datasets"""
        return {
            "min_length": 100,
            "max_length": 100000,
            "max_missing_ratio": 0.1,
            "max_outlier_ratio": 0.05,
            "required_metadata": [
                "source",
                "description",
                "sampling_frequency",
                "units",
                "collection_date"
            ],
            "optional_metadata": [
                "location",
                "instrument",
                "processing_steps",
                "quality_metrics"
            ]
        }
    
    @staticmethod
    def get_validation_checks() -> List[str]:
        """Get validation checks for datasets"""
        return [
            "data_type_validation",
            "range_validation",
            "missing_value_check",
            "outlier_detection",
            "stationarity_test",
            "metadata_completeness",
            "file_integrity"
        ]


class ComplianceChecker:
    """Checker for compliance with submission standards"""
    
    def __init__(self):
        self.model_standards = ModelStandards()
        self.dataset_standards = DatasetStandards()
    
    def check_model_compliance(self, model) -> ValidationResult:
        """Check if a model complies with standards"""
        errors = []
        warnings = []
        
        # Check required methods
        required_methods = ModelStandards.get_required_interface()
        for method in required_methods:
            if not hasattr(model, method):
                errors.append(f"Missing required method: {method}")
        
        # Check if model is fitted
        if hasattr(model, 'fitted') and not model.fitted:
            warnings.append("Model is not fitted")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=errors,
                warnings=warnings
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=errors,
            warnings=warnings
        )
    
    def check_dataset_compliance(self, data: pd.DataFrame) -> ValidationResult:
        """Check if a dataset complies with standards"""
        errors = []
        warnings = []
        
        # Check length
        min_length = DatasetStandards.get_format_requirements()["min_length"]
        if len(data) < min_length:
            errors.append(f"Dataset too short: {len(data)} < {min_length}")
        
        # Check required columns
        required_columns = DatasetStandards.get_format_requirements()["required_columns"]
        for col in required_columns:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=errors,
                warnings=warnings
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=errors,
            warnings=warnings
        )

    def validate_model_interface(self, model_class: type) -> ValidationResult:
        """Validate that a model class meets the required interface"""
        required_methods = ModelStandards.get_required_interface()
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(model_class, method):
                missing_methods.append(method)
        
        if missing_methods:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=[f"Missing required methods: {missing_methods}"],
                warnings=[],
                details={"missing_methods": missing_methods}
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=[],
            warnings=[],
            details={"checked_methods": required_methods}
        )
    
    def validate_model_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate model parameters against constraints"""
        constraints = ModelStandards.get_parameter_constraints()
        issues = []
        
        # Check required parameters
        required_params = constraints["required_parameters"]
        for param in required_params:
            if param not in parameters:
                issues.append(f"Missing required parameter: {param}")
        
        # Check parameter ranges
        if "hurst" in parameters:
            hurst = parameters["hurst"]
            if not (constraints["hurst_range"][0] <= hurst <= constraints["hurst_range"][1]):
                issues.append(f"Hurst exponent {hurst} outside valid range {constraints['hurst_range']}")
        
        if "alpha" in parameters:
            alpha = parameters["alpha"]
            if not (constraints["alpha_range"][0] <= alpha <= constraints["alpha_range"][1]):
                issues.append(f"Alpha {alpha} outside valid range {constraints['alpha_range']}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=issues,
                warnings=[],
                details={"issues": issues}
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=[],
            warnings=[],
            details={"checked_parameters": list(parameters.keys())}
        )
    
    def validate_dataset_format(self, file_path: str) -> ValidationResult:
        """Validate dataset file format"""
        format_requirements = DatasetStandards.get_format_requirements()
        issues = []
        
        # Check file format
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in [f".{fmt}" for fmt in format_requirements["supported_formats"]]:
            issues.append(f"Unsupported file format: {file_ext}")
        
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            issues.append(f"File does not exist: {file_path}")
        elif not os.access(file_path, os.R_OK):
            issues.append(f"File is not readable: {file_path}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=issues,
                warnings=[],
                details={"issues": issues}
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=[],
            warnings=[],
            details={"file_path": file_path, "format": file_ext}
        )
    
    def validate_dataset_content(self, data: pd.DataFrame) -> ValidationResult:
        """Validate dataset content against quality requirements"""
        requirements = DatasetStandards.get_quality_requirements()
        issues = []
        
        # Check data length
        if len(data) < requirements["min_length"]:
            issues.append(f"Data too short: {len(data)} < {requirements['min_length']}")
        elif len(data) > requirements["max_length"]:
            issues.append(f"Data too long: {len(data)} > {requirements['max_length']}")
        
        # Check required columns
        required_cols = DatasetStandards.get_format_requirements()["required_columns"]
        for col in required_cols:
            if col not in data.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check missing values
        if "value" in data.columns:
            missing_ratio = data["value"].isnull().sum() / len(data)
            if missing_ratio > requirements["max_missing_ratio"]:
                issues.append(f"Too many missing values: {missing_ratio:.3f} > {requirements['max_missing_ratio']}")
        
        # Check for infinite values
        if "value" in data.columns:
            inf_count = np.isinf(data["value"]).sum()
            if inf_count > 0:
                issues.append(f"Found {inf_count} infinite values")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=issues,
                warnings=[],
                details={"issues": issues}
            )
        
        return ValidationResult(
            is_valid=True,
            status=SubmissionStatus.APPROVED,
            errors=[],
            warnings=[],
            details={"data_length": len(data), "columns": list(data.columns)}
        )
