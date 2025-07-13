"""
Base classes and interfaces for the pipeline architecture.

This module provides the foundation for building modular, reusable pipelines
for processing data through multiple stages.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TypeVar, Generic
import time
from datetime import datetime

from src.utils.logger import get_logger


T = TypeVar('T')


@dataclass
class PipelineData:
    """
    Container for data flowing through the pipeline.
    
    Attributes:
        raw_input: The original input data
        processed_data: Data being processed through stages
        metadata: Stage-specific metadata and results
        context: Shared context information
        timestamps: Timing information for each stage
    """
    raw_input: Any
    processed_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, float] = field(default_factory=dict)
    
    def add_stage_result(self, stage_name: str, key: str, value: Any):
        """Add a result from a specific stage."""
        if stage_name not in self.processed_data:
            self.processed_data[stage_name] = {}
        self.processed_data[stage_name][key] = value
    
    def get_stage_result(self, stage_name: str, key: str, default=None):
        """Get a result from a specific stage."""
        return self.processed_data.get(stage_name, {}).get(key, default)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the pipeline data."""
        self.metadata[key] = value
    
    def record_timestamp(self, stage_name: str, timestamp: float):
        """Record processing time for a stage."""
        self.timestamps[stage_name] = timestamp


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage processes data and passes it to the next stage.
    """
    
    def __init__(self, name: str = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline stage."""
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = get_logger(self.name)
        self._is_initialized = False
    
    async def initialize(self):
        """
        Initialize the stage (load models, connect to services, etc.).
        
        This method is called once before the first process call.
        """
        if not self._is_initialized:
            await self._initialize()
            self._is_initialized = True
    
    @abstractmethod
    async def _initialize(self):
        """Stage-specific initialization logic."""
        pass
    
    @abstractmethod
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Process the pipeline data.
        
        Args:
            data: The pipeline data to process
            
        Returns:
            The processed pipeline data
        """
        pass
    
    async def validate_input(self, data: PipelineData) -> bool:
        """
        Validate input data before processing.
        
        Override this method to add stage-specific validation.
        
        Args:
            data: The pipeline data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    async def handle_error(self, error: Exception, data: PipelineData) -> PipelineData:
        """
        Handle errors during processing.
        
        Override this method to add stage-specific error handling.
        
        Args:
            error: The exception that occurred
            data: The pipeline data being processed
            
        Returns:
            The pipeline data (possibly modified)
        """
        self.logger.error(f"Error in stage {self.name}: {str(error)}")
        data.add_metadata(f"{self.name}_error", str(error))
        raise error


class Pipeline:
    """
    Orchestrator for running data through multiple pipeline stages.
    """
    
    def __init__(self, stages: List[PipelineStage], name: str = "Pipeline"):
        """
        Initialize the pipeline with stages.
        
        Args:
            stages: List of pipeline stages to execute in order
            name: Name of the pipeline
        """
        self.stages = stages
        self.name = name
        self.logger = get_logger(name)
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize all pipeline stages."""
        if not self._is_initialized:
            self.logger.info(f"Initializing pipeline '{self.name}' with {len(self.stages)} stages")
            
            for stage in self.stages:
                self.logger.debug(f"Initializing stage: {stage.name}")
                await stage.initialize()
            
            self._is_initialized = True
            self.logger.info(f"Pipeline '{self.name}' initialized successfully")
    
    async def process(self, 
                     input_data: Any, 
                     context: Optional[Dict[str, Any]] = None) -> PipelineData:
        """
        Process input data through all pipeline stages.
        
        Args:
            input_data: The input data to process
            context: Optional context information
            
        Returns:
            The final pipeline data after all stages
        """
        # Ensure pipeline is initialized
        await self.initialize()
        
        # Create pipeline data container
        data = PipelineData(
            raw_input=input_data,
            context=context or {}
        )
        
        # Record start time
        start_time = time.time()
        data.add_metadata('pipeline_start_time', start_time)
        
        # Process through each stage
        for i, stage in enumerate(self.stages):
            stage_start = time.time()
            
            try:
                # Validate input
                if not await stage.validate_input(data):
                    raise ValueError(f"Invalid input for stage {stage.name}")
                
                # Process data
                self.logger.debug(f"Processing stage {i+1}/{len(self.stages)}: {stage.name}")
                data = await stage.process(data)
                
                # Record timing
                stage_duration = (time.time() - stage_start) * 1000
                data.record_timestamp(stage.name, stage_duration)
                
            except Exception as e:
                self.logger.error(f"Error in stage {stage.name}: {str(e)}")
                
                # Try to handle error
                try:
                    data = await stage.handle_error(e, data)
                except Exception:
                    # If error handling fails, stop pipeline
                    data.add_metadata('failed_stage', stage.name)
                    data.add_metadata('pipeline_error', str(e))
                    break
        
        # Record total time
        total_time = (time.time() - start_time) * 1000
        data.add_metadata('pipeline_total_time_ms', total_time)
        
        self.logger.info(f"Pipeline '{self.name}' completed in {total_time:.2f}ms")
        
        return data
    
    def add_stage(self, stage: PipelineStage, index: Optional[int] = None):
        """
        Add a stage to the pipeline.
        
        Args:
            stage: The stage to add
            index: Position to insert the stage (None for end)
        """
        if index is None:
            self.stages.append(stage)
        else:
            self.stages.insert(index, stage)
        
        # Reset initialization flag
        self._is_initialized = False
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove a stage from the pipeline.
        
        Args:
            stage_name: Name of the stage to remove
            
        Returns:
            True if stage was removed, False if not found
        """
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                self.stages.pop(i)
                self._is_initialized = False
                return True
        return False
    
    def get_stage_names(self) -> List[str]:
        """Get names of all stages in the pipeline."""
        return [stage.name for stage in self.stages]


class ConditionalStage(PipelineStage):
    """
    A pipeline stage that conditionally executes based on a condition.
    """
    
    def __init__(self, 
                 stage: PipelineStage, 
                 condition_fn,
                 name: str = None):
        """
        Initialize conditional stage.
        
        Args:
            stage: The stage to conditionally execute
            condition_fn: Function that takes PipelineData and returns bool
            name: Optional name for the conditional stage
        """
        super().__init__(name or f"Conditional_{stage.name}")
        self.stage = stage
        self.condition_fn = condition_fn
    
    async def _initialize(self):
        """Initialize the wrapped stage."""
        await self.stage.initialize()
    
    async def process(self, data: PipelineData) -> PipelineData:
        """Process data if condition is met."""
        if self.condition_fn(data):
            self.logger.debug(f"Condition met, executing stage: {self.stage.name}")
            return await self.stage.process(data)
        else:
            self.logger.debug(f"Condition not met, skipping stage: {self.stage.name}")
            return data


class ParallelStage(PipelineStage):
    """
    A pipeline stage that executes multiple stages in parallel.
    """
    
    def __init__(self, 
                 stages: List[PipelineStage], 
                 merge_fn=None,
                 name: str = "ParallelStage"):
        """
        Initialize parallel stage.
        
        Args:
            stages: List of stages to execute in parallel
            merge_fn: Optional function to merge results
            name: Name for the parallel stage
        """
        super().__init__(name)
        self.stages = stages
        self.merge_fn = merge_fn
    
    async def _initialize(self):
        """Initialize all parallel stages."""
        await asyncio.gather(*[stage.initialize() for stage in self.stages])
    
    async def process(self, data: PipelineData) -> PipelineData:
        """Process data through all stages in parallel."""
        # Create copies of data for each stage
        stage_data = [PipelineData(
            raw_input=data.raw_input,
            processed_data=data.processed_data.copy(),
            metadata=data.metadata.copy(),
            context=data.context.copy(),
            timestamps=data.timestamps.copy()
        ) for _ in self.stages]
        
        # Process in parallel
        results = await asyncio.gather(
            *[stage.process(stage_data[i]) for i, stage in enumerate(self.stages)]
        )
        
        # Merge results
        if self.merge_fn:
            return self.merge_fn(results)
        else:
            # Default merge: combine all processed_data
            merged_data = data
            for i, result in enumerate(results):
                stage_name = self.stages[i].name
                merged_data.processed_data[f"parallel_{stage_name}"] = result.processed_data
            
            return merged_data