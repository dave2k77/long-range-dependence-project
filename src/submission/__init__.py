"""
Submission Module for Long-Range Dependence Project

This module provides functionality for users to submit new estimator models and datasets
to the benchmark. It includes validation, testing, and integration with the full analysis pipeline.
"""

from .model_submission import (
    ModelSubmission,
    ModelValidator,
    ModelTester,
    ModelRegistry
)

from .dataset_submission import (
    DatasetSubmission,
    DatasetValidator,
    DatasetTester,
    DatasetRegistry
)

from .submission_manager import (
    SubmissionManager,
    SubmissionResult,
    SubmissionStatus,
    process_submission
)

from .standards import (
    ModelStandards,
    DatasetStandards,
    ValidationResult,
    ComplianceChecker
)

__all__ = [
    # Model Submission
    'ModelSubmission',
    'ModelValidator',
    'ModelTester',
    'ModelRegistry',
    
    # Dataset Submission
    'DatasetSubmission',
    'DatasetValidator',
    'DatasetTester',
    'DatasetRegistry',
    
    # Submission Management
    'SubmissionManager',
    'SubmissionResult',
    'SubmissionStatus',
    'process_submission',
    
    # Standards
    'ModelStandards',
    'DatasetStandards',
    'ValidationResult',
    'ComplianceChecker'
]
