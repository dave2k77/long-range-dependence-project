"""
Submission Manager Module

This module coordinates the entire submission process and integrates submitted
models and datasets with the full analysis pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import shutil

from .model_submission import ModelSubmission, ModelMetadata
from .dataset_submission import DatasetSubmission, DatasetMetadata
from .standards import SubmissionStatus, ValidationResult


@dataclass
class SubmissionResult:
    """Result of a submission process"""
    submission_id: str
    submission_type: str  # "model" or "dataset"
    status: SubmissionStatus
    success: bool
    message: str
    validation_results: List[ValidationResult]
    test_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    integration_results: Optional[Dict[str, Any]] = None
    submission_date: Optional[str] = None
    processing_time: Optional[float] = None
    errors: Optional[List[str]] = None


class SubmissionManager:
    """Main manager for handling all submissions"""
    
    def __init__(self, 
                 models_registry_path: str = "models/registry.json",
                 datasets_registry_path: str = "datasets/registry.json",
                 results_dir: str = "results/submissions"):
        self.model_submission = ModelSubmission(models_registry_path)
        self.dataset_submission = DatasetSubmission(datasets_registry_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Track submissions
        self.submissions = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for submission manager"""
        logger = logging.getLogger("SubmissionManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = self.results_dir / "submission_manager.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def submit_model(self, 
                    model_file: str,
                    metadata: ModelMetadata,
                    run_full_analysis: bool = True) -> SubmissionResult:
        """Submit a new model for validation, testing, and integration"""
        submission_id = f"model_{metadata.name}_{metadata.version}_{int(time.time())}"
        
        self.logger.info(f"Starting model submission: {submission_id}")
        
        result = SubmissionResult(
            submission_id=submission_id,
            submission_type="model",
            status=SubmissionStatus.VALIDATING,
            success=False,
            message="",
            validation_results=[],
            submission_date=datetime.now().isoformat()
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Submit model for validation and testing
            submission_response = self.model_submission.submit_model(
                model_file=model_file,
                metadata=metadata,
                test_model=True
            )
            
            result.validation_results = submission_response.get("validation_results", [])
            result.test_results = submission_response.get("test_results")
            result.performance_metrics = submission_response.get("performance_evaluation")
            
            if not submission_response["success"]:
                result.status = SubmissionStatus.REJECTED
                result.message = submission_response["message"]
                result.errors = [submission_response["message"]]
                return result
            
            # Step 2: Run full analysis if requested
            if run_full_analysis and submission_response.get("model_id"):
                result.status = SubmissionStatus.TESTING
                
                integration_results = self._run_full_analysis_with_model(
                    submission_response["model_id"],
                    metadata
                )
                result.integration_results = integration_results
                
                if integration_results.get("success", False):
                    result.status = SubmissionStatus.APPROVED
                    result.message = "Model submitted and integrated successfully"
                else:
                    result.status = SubmissionStatus.REJECTED
                    result.message = "Model failed integration testing"
                    result.errors = integration_results.get("errors", [])
            else:
                result.status = SubmissionStatus.APPROVED
                result.message = "Model submitted successfully"
            
            result.success = result.status == SubmissionStatus.APPROVED
            
        except Exception as e:
            result.status = SubmissionStatus.REJECTED
            result.message = f"Submission failed: {str(e)}"
            result.errors = [str(e)]
            self.logger.error(f"Model submission failed: {str(e)}")
        
        result.processing_time = time.time() - start_time
        
        # Save submission result
        self._save_submission_result(result)
        
        self.logger.info(f"Model submission completed: {submission_id} - Status: {result.status.value}")
        
        return result
    
    def submit_dataset(self, 
                      file_path: str,
                      metadata: DatasetMetadata,
                      run_full_analysis: bool = True) -> SubmissionResult:
        """Submit a new dataset for validation, testing, and integration"""
        submission_id = f"dataset_{metadata.name}_{metadata.version}_{int(time.time())}"
        
        self.logger.info(f"Starting dataset submission: {submission_id}")
        
        result = SubmissionResult(
            submission_id=submission_id,
            submission_type="dataset",
            status=SubmissionStatus.VALIDATING,
            success=False,
            message="",
            validation_results=[],
            submission_date=datetime.now().isoformat()
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Submit dataset for validation and testing
            submission_response = self.dataset_submission.submit_dataset(
                file_path=file_path,
                metadata=metadata,
                test_dataset=True
            )
            
            result.validation_results = submission_response.get("validation_results", [])
            result.test_results = submission_response.get("test_results")
            result.performance_metrics = submission_response.get("quality_evaluation")
            
            if not submission_response["success"]:
                result.status = SubmissionStatus.REJECTED
                result.message = submission_response["message"]
                result.errors = [submission_response["message"]]
                return result
            
            # Step 2: Run full analysis if requested
            if run_full_analysis and submission_response.get("dataset_id"):
                result.status = SubmissionStatus.TESTING
                
                integration_results = self._run_full_analysis_with_dataset(
                    submission_response["dataset_id"],
                    metadata
                )
                result.integration_results = integration_results
                
                if integration_results.get("success", False):
                    result.status = SubmissionStatus.APPROVED
                    result.message = "Dataset submitted and integrated successfully"
                else:
                    result.status = SubmissionStatus.REJECTED
                    result.message = "Dataset failed integration testing"
                    result.errors = integration_results.get("errors", [])
            else:
                result.status = SubmissionStatus.APPROVED
                result.message = "Dataset submitted successfully"
            
            result.success = result.status == SubmissionStatus.APPROVED
            
        except Exception as e:
            result.status = SubmissionStatus.REJECTED
            result.message = f"Submission failed: {str(e)}"
            result.errors = [str(e)]
            self.logger.error(f"Dataset submission failed: {str(e)}")
        
        result.processing_time = time.time() - start_time
        
        # Save submission result
        self._save_submission_result(result)
        
        self.logger.info(f"Dataset submission completed: {submission_id} - Status: {result.status.value}")
        
        return result
    
    def _run_full_analysis_with_model(self, model_id: str, metadata: ModelMetadata) -> Dict[str, Any]:
        """Run full analysis pipeline with the submitted model"""
        integration_results = {
            "success": False,
            "analysis_results": {},
            "comparison_results": {},
            "errors": []
        }
        
        try:
            # Import the submitted model
            model_file = metadata.file_path
            spec = __import__('importlib.util').spec_from_file_location("submitted_model", model_file)
            module = __import__('importlib.util').module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the model class
            model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'fit') and 
                    hasattr(attr, 'estimate_hurst')):
                    model_class = attr
                    break
            
            if model_class is None:
                integration_results["errors"].append("Could not find valid model class")
                return integration_results
            
            # Load test datasets
            test_datasets = self._load_test_datasets()
            
            # Run analysis with the submitted model
            for dataset_name, dataset_data in test_datasets.items():
                try:
                    # Initialize model
                    model = model_class(**metadata.parameters)
                    
                    # Run analysis
                    model.fit(dataset_data)
                    hurst_est = model.estimate_hurst()
                    alpha_est = model.estimate_alpha()
                    
                    integration_results["analysis_results"][dataset_name] = {
                        "hurst_estimate": hurst_est,
                        "alpha_estimate": alpha_est,
                        "success": True
                    }
                    
                except Exception as e:
                    integration_results["analysis_results"][dataset_name] = {
                        "error": str(e),
                        "success": False
                    }
                    integration_results["errors"].append(f"Analysis failed for {dataset_name}: {str(e)}")
            
            # Compare with existing models
            comparison_results = self._compare_with_existing_models(
                model_class, metadata.parameters, test_datasets
            )
            integration_results["comparison_results"] = comparison_results
            
            # Determine success
            successful_analyses = sum(
                1 for result in integration_results["analysis_results"].values()
                if result.get("success", False)
            )
            total_analyses = len(integration_results["analysis_results"])
            
            if successful_analyses / total_analyses >= 0.8:  # 80% success rate
                integration_results["success"] = True
            
        except Exception as e:
            integration_results["errors"].append(f"Integration failed: {str(e)}")
        
        return integration_results
    
    def _run_full_analysis_with_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> Dict[str, Any]:
        """Run full analysis pipeline with the submitted dataset"""
        integration_results = {
            "success": False,
            "analysis_results": {},
            "model_comparison": {},
            "errors": []
        }
        
        try:
            # Load the submitted dataset
            data = pd.read_csv(metadata.file_path)
            if "value" not in data.columns:
                integration_results["errors"].append("Dataset missing 'value' column")
                return integration_results
            
            dataset_values = data["value"].dropna().values
            
            if len(dataset_values) < 100:
                integration_results["errors"].append("Dataset too short for analysis")
                return integration_results
            
            # Run analysis with existing models
            existing_models = self._get_existing_models()
            
            for model_name, model_info in existing_models.items():
                try:
                    # Import and run the model
                    model_result = self._run_model_on_dataset(model_name, model_info, dataset_values)
                    integration_results["analysis_results"][model_name] = model_result
                    
                except Exception as e:
                    integration_results["analysis_results"][model_name] = {
                        "error": str(e),
                        "success": False
                    }
                    integration_results["errors"].append(f"Model {model_name} failed: {str(e)}")
            
            # Compare results across models
            comparison = self._compare_model_results(integration_results["analysis_results"])
            integration_results["model_comparison"] = comparison
            
            # Determine success
            successful_analyses = sum(
                1 for result in integration_results["analysis_results"].values()
                if result.get("success", False)
            )
            total_analyses = len(integration_results["analysis_results"])
            
            if successful_analyses / total_analyses >= 0.7:  # 70% success rate
                integration_results["success"] = True
            
        except Exception as e:
            integration_results["errors"].append(f"Integration failed: {str(e)}")
        
        return integration_results
    
    def _load_test_datasets(self) -> Dict[str, np.ndarray]:
        """Load test datasets for model validation"""
        # Generate synthetic test datasets
        np.random.seed(42)
        
        test_datasets = {}
        
        # White noise
        test_datasets["white_noise"] = np.random.normal(0, 1, 1000)
        
        # Fractional Brownian motion with different Hurst exponents
        for hurst in [0.3, 0.5, 0.7, 0.9]:
            test_datasets[f"fbm_h{int(hurst*10)}"] = self._generate_fbm(1000, hurst)
        
        # AR(1) process
        test_datasets["ar1"] = self._generate_ar1(1000, 0.8)
        
        return test_datasets
    
    def _generate_fbm(self, length: int, hurst: float) -> np.ndarray:
        """Generate fractional Brownian motion"""
        d = hurst - 0.5
        n = length
        
        fgn = np.zeros(n)
        for i in range(1, n):
            fgn[i] = fgn[i-1] + np.random.normal(0, 1) * (i ** d)
        
        return fgn
    
    def _generate_ar1(self, length: int, phi: float) -> np.ndarray:
        """Generate AR(1) process"""
        x = np.zeros(length)
        x[0] = np.random.normal(0, 1)
        
        for i in range(1, length):
            x[i] = phi * x[i-1] + np.random.normal(0, 1)
        
        return x
    
    def _get_existing_models(self) -> Dict[str, Any]:
        """Get list of existing models for comparison"""
        # This would typically load from the model registry
        # For now, return a simple structure
        return {
            "dfa": {"type": "builtin", "module": "src.analysis.dfa_analysis"},
            "rs": {"type": "builtin", "module": "src.analysis.rs_analysis"},
            "wavelet": {"type": "builtin", "module": "src.analysis.wavelet_analysis"},
            "spectral": {"type": "builtin", "module": "src.analysis.spectral_analysis"}
        }
    
    def _run_model_on_dataset(self, model_name: str, model_info: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """Run a specific model on the dataset"""
        try:
            # Import the model module
            module = __import__(model_info["module"], fromlist=["*"])
            
            # Run analysis based on model type
            if model_name == "dfa":
                from src.analysis.dfa_analysis import dfa
                result = dfa(data)
                return {
                    "hurst_estimate": result.get("hurst_exponent", 0.5),
                    "alpha_estimate": result.get("alpha", 1.0),
                    "r_squared": result.get("r_squared", 0.0),
                    "success": True
                }
            
            elif model_name == "rs":
                from src.analysis.rs_analysis import rs_analysis
                result = rs_analysis(data)
                return {
                    "hurst_estimate": result.get("hurst_exponent", 0.5),
                    "alpha_estimate": result.get("alpha", 1.0),
                    "r_squared": result.get("r_squared", 0.0),
                    "success": True
                }
            
            elif model_name == "wavelet":
                from src.analysis.wavelet_analysis import wavelet_whittle_estimation
                result = wavelet_whittle_estimation(data)
                return {
                    "hurst_estimate": result.get("hurst_exponent", 0.5),
                    "alpha_estimate": result.get("alpha", 1.0),
                    "r_squared": result.get("r_squared", 0.0),
                    "success": True
                }
            
            elif model_name == "spectral":
                from src.analysis.spectral_analysis import whittle_mle
                result = whittle_mle(data)
                return {
                    "hurst_estimate": result.get("hurst_exponent", 0.5),
                    "alpha_estimate": result.get("alpha", 1.0),
                    "r_squared": result.get("r_squared", 0.0),
                    "success": True
                }
            
            else:
                return {
                    "error": f"Unknown model: {model_name}",
                    "success": False
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _compare_with_existing_models(self, submitted_model_class: type, parameters: Dict[str, Any], test_datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compare submitted model with existing models"""
        comparison = {
            "performance_comparison": {},
            "accuracy_comparison": {},
            "speed_comparison": {}
        }
        
        # This would implement detailed comparison logic
        # For now, return a simple structure
        comparison["performance_comparison"] = {
            "submitted_model": "Good",
            "existing_models": "Good"
        }
        
        return comparison
    
    def _compare_model_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across different models"""
        comparison = {
            "hurst_estimates": {},
            "consistency": {},
            "recommendations": []
        }
        
        # Extract Hurst estimates
        hurst_estimates = {}
        for model_name, result in analysis_results.items():
            if result.get("success", False):
                hurst_estimates[model_name] = result.get("hurst_estimate", 0.5)
        
        comparison["hurst_estimates"] = hurst_estimates
        
        # Calculate consistency
        if len(hurst_estimates) > 1:
            values = list(hurst_estimates.values())
            std_dev = np.std(values)
            mean_val = np.mean(values)
            
            comparison["consistency"] = {
                "mean_hurst": float(mean_val),
                "std_dev": float(std_dev),
                "coefficient_of_variation": float(std_dev / mean_val) if mean_val != 0 else float('inf')
            }
            
            # Generate recommendations
            if std_dev > 0.1:
                comparison["recommendations"].append("High variability in Hurst estimates across models")
            if mean_val < 0.4 or mean_val > 0.9:
                comparison["recommendations"].append("Hurst estimate outside typical range")
        
        return comparison
    
    def _save_submission_result(self, result: SubmissionResult):
        """Save submission result to file"""
        result_file = self.results_dir / f"{result.submission_id}.json"
        
        # Convert to dict for JSON serialization
        result_dict = asdict(result)
        result_dict["status"] = result.status.value
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        # Also save to submissions tracking
        self.submissions[result.submission_id] = result
    
    def get_submission_status(self, submission_id: str) -> Optional[SubmissionStatus]:
        """Get the status of a submission"""
        if submission_id in self.submissions:
            return self.submissions[submission_id].status
        return None
    
    def list_submissions(self, submission_type: Optional[str] = None) -> List[str]:
        """List all submissions, optionally filtered by type"""
        if submission_type is None:
            return list(self.submissions.keys())
        else:
            return [
                sub_id for sub_id, result in self.submissions.items()
                if result.submission_type == submission_type
            ]
    
    def get_submission_result(self, submission_id: str) -> Optional[SubmissionResult]:
        """Get the full result of a submission"""
        return self.submissions.get(submission_id)


def process_submission(submission_type: str, **kwargs) -> SubmissionResult:
    """Convenience function to process a submission"""
    manager = SubmissionManager()
    
    if submission_type == "model":
        return manager.submit_model(**kwargs)
    elif submission_type == "dataset":
        return manager.submit_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown submission type: {submission_type}")
